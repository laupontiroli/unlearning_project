# `model.py`, line by line

Laura asked me to write `model.py` from an article, so PLAN §11 rule 3 applies: I wrote it, and this file walks through it.

**Source.** Andrej Karpathy's nanoGPT `model.py` (github.com/karpathy/nanoGPT, MIT License), which is the code from his video and article *"Let's build GPT: from scratch, in code, spelled out"*. The structure is the same: CausalSelfAttention → MLP → Block → GPT, a fused QKV projection, pre-LayerNorm, weight tying, GPT-2 initialization and the `generate` loop. The changes I made are listed at the end.

Shapes: B = batch, T = sequence length, C = d_model = 256, H = 4 heads, D = C/H = 64, V = 62.

---

## 1. `causal_attention_by_hand(q, k, v)`

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(D)      # (B,H,T,T)
```
- Each position t asks a question (query q_t), and each position s offers a label (key k_s).
- Their dot product is a score saying how relevant s is to t.
- Dividing by √D keeps the scores' variance around 1. Without it, the softmax becomes nearly one-hot when D is large and gradients vanish.

```python
mask = torch.ones(T, T, dtype=torch.bool).tril()    # lower triangle: s <= t
att = att.masked_fill(~mask, float("-inf"))
```
- **The causal mask.** Position t may only look at s ≤ t. Above the diagonal, s is in the future, which is exactly the character the model has to predict.
- Setting those scores to −∞ makes their softmax weight exp(−∞) = 0.
- Without the mask the model would copy the answer: training loss would go to almost 0 and generation would fail. `test_causal_mask` checks for this.

```python
att = F.softmax(att, dim=-1)       # each row sums to 1 -> weights over past positions
att = F.dropout(att, p, training)  # randomly drop attention links (regularization), training only
return att @ v                     # weighted average of the values: (B,H,T,T)@(B,H,T,D) -> (B,H,T,D)
```
The output at t is a mixture of the value vectors of positions 0…t, weighted by relevance.

`F.scaled_dot_product_attention(..., is_causal=True)` computes the same thing in a fused, faster kernel. `test_attention_by_hand_matches_sdpa` checks that the two agree to within 1e-5.

## 2. `CausalSelfAttention`

```python
self.qkv  = nn.Linear(C, 3C)   # one matmul gives Q, K and V for all heads
self.proj = nn.Linear(C, C)    # mixes the heads' outputs back into the residual stream
```
In `forward`:
1. `split(C, dim=2)` cuts the 3C output into Q, K and V, each (B,T,C).
2. `.view(B,T,H,D).transpose(1,2)` → (B,H,T,D). The 256 channels become 4 independent heads of 64, so each head can attend to different things (for example, one tracks the previous letter and another tracks word starts).
3. Attention runs with either SDPA or the by-hand version (`cfg.use_sdpa`). Dropout is used only in `self.training` mode.
4. `.transpose(1,2).contiguous().view(B,T,C)` glues the heads back side by side.
5. `proj` then dropout gives what is added to the residual stream.

## 3. `MLP`

`Linear(C,4C) → GELU → Linear(4C,C) → dropout`. Attention moves information *between* positions; the MLP processes it *at* each position on its own. Expanding to 4C (1024) gives room for features, and GELU is the non-linearity, a smooth version of ReLU.

## 4. `Block` (pre-LayerNorm)

```python
x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```
- **Residual connections** (`x + …`): each sub-layer adds a correction instead of replacing x. Gradients flow straight through the `+`, so 6 layers train easily.
- **Pre-LN**: normalizing *before* each sub-layer (GPT-2 style) keeps the inputs of attention and MLP at a stable scale. It trains more stably than the original post-LN design.

## 5. `GPT.__init__`

```python
self.tok_emb = nn.Embedding(V, C)            # character id -> 256-d vector
self.pos_emb = nn.Embedding(block_size, C)   # position 0..511 -> 256-d vector (learned, absolute)
self.blocks  = ModuleList([Block] * 6)
self.ln_f    = LayerNorm(C)                  # final norm before the output head
self.head    = Linear(C, V, bias=False)
self.head.weight = self.tok_emb.weight       # WEIGHT TYING
```
- Attention by itself ignores order (it treats the sequence as a set), so position embeddings tell the model where each character is.
- **Weight tying.** The same (V × C) matrix turns characters into vectors on the way in and vectors into character scores on the way out (score = h · e_c). This saves V·C parameters, and characters that behave alike get similar vectors on both sides. `num_params()` counts the shared matrix once, because `parameters()` removes duplicates.

**Initialization** (GPT-2):
- `_init_weights`: every Linear and Embedding weight ~ N(0, 0.02), biases 0. The small weights keep the initial logits close to 0, so the initial loss ≈ ln V.
- Then every `*.proj.weight` (attention output and MLP output) is re-drawn with std 0.02/√(2·n_layer). Each block adds 2 terms to the residual stream (2·n_layer in total), and this scaling stops the variance of x from growing with depth.

## 6. `GPT.forward`

```python
pos = torch.arange(T)
x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B,T,C) + (T,C) broadcasts over B
for block in self.blocks: x = block(x)
logits = self.head(self.ln_f(x))                       # (B,T,V): scores for the next char at every t
loss = F.cross_entropy(logits.view(-1,V), targets.reshape(-1))
```
- `cross_entropy` = mean over all B·T positions of −log p(correct next character), in **nats**.
- bpc = loss / ln 2.
- Each training window therefore gives T = 512 prediction problems at once. This is why the causal mask matters.

## 7. `GPT.generate`

Each step:
1. Crop the context to the last 512 characters, the most the position embeddings cover.
2. Run the model and take the logits of the **last** position only.
3. Divide by `temperature`: below 1 is safer and more repetitive, above 1 is more random.
4. With `top_k`, set every logit outside the k best to −∞.
5. Apply softmax, then `torch.multinomial(..., generator=g)`. Passing a seeded generator makes the samples reproducible (§9.2 asks for a fixed seed per prompt).
6. Append the sampled character and repeat.

---

## Changes from nanoGPT, and why

| nanoGPT | here | why |
|---|---|---|
| auto flash detection plus a manual branch | stand-alone `causal_attention_by_hand` and a `use_sdpa` flag | PLAN §5 asks for attention by hand, checked against SDPA |
| at inference, logits only for the last position | logits for **all** positions | evaluation needs every position's distribution (bpc, KL to θ_ref) |
| `generate` without a generator | `generator=` argument | reproducible samples per prompt |
| optional `bias=False`, custom LayerNorm | biases on, `nn.LayerNorm` | simpler. Costs +20k parameters: 4,886,016 vs the plan's formula 4,865,536 (+0.42%, inside the 1% test) |
| `c_attn` / `c_fc` / `c_proj` | `qkv` / `fc` / `proj` | names match the original stubs |

## Gate 1 numbers (`scripts/model_check.py` → `results/gate1.json`)

- Parameters: **4,886,016**, which is 0.42% above the formula.
- Initial loss on a real PT batch: **4.20 nats** (6.06 bpc), against ln 62 = 4.13 (5.95 bits). That is close to uniform, as expected from the small initialization.
- Overfitting one batch (4 × 128 characters, dropout 0): the loss falls from 4.20 to **0.003** in 300 steps (4.5 s on MPS). The curve is in `results/figures/overfit_curve.png`.
- An untrained sample is gibberish, as it should be: `o brasil é 'm66d2çcdéhrre1e2w1,,íí)àààké1dimsô…`

## Questions to be able to answer
1. What happens to training loss and to generation if you delete the mask line?
2. Why divide by √D?
3. Why is the initial loss ≈ ln V, and what would 6.06 vs 5.95 bits mean if it were much larger?
4. What does weight tying share, and why does `num_params` not count it twice?
5. Why is only the last position's logit used in `generate`, while training uses all of them?
