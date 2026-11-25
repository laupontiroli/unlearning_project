# PLAN: Language-level unlearning in a character-level transformer
**Portuguese-only model → add English → unlearn English → compare with the model that never saw English**

Course: NLP final project, Researcher track. Report: 2 pages, IEEE conference template. Machine: MacBook Air M4, 16 GB (PyTorch MPS), long runs on Colab/Kaggle GPU.

---

## 0. How to use this file

- **Laura:** paragraphs marked **Why** explain the reasoning so you can defend it in the interview. §12 lists the concepts and questions you must be able to answer. Do not hand over understanding together with the code.
- **Claude Code:** read §1–§3 and §11 first. Then execute §4–§9 in order. Every phase ends with a **STOP gate**: print the requested outputs, wait for Laura's approval, then continue. Never skip a gate.

---

## 1. Goal and research question

**Research question.** A character-level transformer is trained only on Portuguese (the *reference* model). It is then trained further on English (the *bilingual* model). Can gradient-based unlearning that uses **only** the bilingual model, an English forget set and a Portuguese retain set bring behaviour back to the reference, measured by:

1. bits per character on held-out Portuguese and English text,
2. how much English leaks into generated text,
3. the distance between next-character distributions of the unlearned model and the reference,
4. resistance to *relearning* (how quickly English comes back with little data)?

**Hypotheses (to test, not assume).**
- **H1.** Plain gradient ascent (GA) on English wrecks Portuguese before English reaches the reference's level.
- **H2.** Gradient ascent plus gradient descent on Portuguese (NegGrad+) gets English bits-per-character close to the reference while keeping Portuguese, but relearning on small amounts of English recovers faster than it does for the reference (knowledge is dormant, not gone).
- **H3.** A control that only fine-tunes on Portuguese (no ascent) forgets less English than NegGrad+. If it does not, ascent is unnecessary.

**Definitions.**

| Name | Meaning |
|---|---|
| θ_ref | Stage-1 model trained on Portuguese only. Saved once, read-only. **Used only for evaluation.** |
| θ_b | θ_ref further trained on Portuguese + English (mixed). The model to unlearn from. Primary experiment. |
| θ_en | θ_ref further trained on English only, short run. Diagnostic: shows the "text stops making sense" effect. |
| Forget set | English training text (`en_forget_train`). |
| Retain set | Portuguese training text (`pt_train`). |
| θ_u | Result of an unlearning method applied to θ_b. |

**Why (Laura).** In classic unlearning the gold standard is "retrain without the forget data". Here English is added *after* Portuguese, so the gold standard is simply the Portuguese checkpoint. This gives an exact reference without retraining. The rule that matters: **the unlearning algorithm must never see θ_ref**. If it did, it would be copying the answer.

---

## 2. Fixed decisions

| Decision | Choice |
|---|---|
| Model type | Decoder-only transformer, character level |
| Corpus | Hugging Face `wikimedia/wikipedia`, configs `20231101.pt` (about 1.11M articles) and `20231101.en` (about 6.41M articles); fields `id`, `url`, `title`, `text` |
| Tokenizer | None. Character vocabulary built by hand (§4) |
| Model code | Written by Laura (see §11 rule 3) |
| Primary stage 2 | Mixed PT + EN (50/50 per batch). Pure-EN is a short diagnostic |
| Seeds | 3 if compute allows (0, 1, 2). Develop everything on seed 0 first |
| Tuning data | Validation splits only. Test splits are touched once at the end |

---

## 3. Environment and repository

**Tooling.** Use `uv` for Python. Python 3.11 or newer. Read and write all text files with explicit `encoding="utf-8"` (Portuguese accents are a recurring pitfall).

```
uv init unlearn-lang && cd unlearn-lang
uv add torch numpy datasets pyarrow pyyaml matplotlib pytest tqdm
```

**Layout.**

```
configs/        data.yaml  model.yaml  stage1.yaml  stage2.yaml  unlearn.yaml  relearn.yaml
data/           raw/ (cache)  processed/*.bin  vocab.json  stats.json  wordlists.json
src/unlearnlang/
    data.py         cleaning, filtering, splitting, packing, batch sampler
    model.py        Laura writes this (CausalSelfAttention, Block, GPT)
    train.py        generic train loop (stage 1 and stage 2)
    unlearn.py      GA, NegGrad+, retain-only control, NPO (stretch)
    evaluate.py     bpc, KL, generation + leakage, relearning
    utils.py        seeding, device, checkpoint IO, logging
scripts/        prepare_data.py  train.py  stage2.py  unlearn.py  evaluate.py  relearn.py  plots.py
tests/          test_data.py  test_model.py  test_isolation.py
checkpoints/    ref_s{seed}.pt (read-only)  bilingual_s{seed}.pt  ...
results/        *.jsonl  *.json  figures/
logs/
```

**Devices.**
- Pick automatically: CUDA, else MPS, else CPU.
- MPS (Mac): use float32 and set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Do not use `torch.compile`.
- CUDA (Colab T4): fp16 autocast with GradScaler. If the GPU supports bf16 (A100, L4), use bf16.
- Save checkpoints as CPU state_dicts so they load on any device.

**Long runs.** Scripts take `--config` and `--seed`, log to `logs/`, checkpoint regularly and resume with `--resume`. On the Mac run them in the background (e.g. `nohup uv run python scripts/train.py ... > logs/x.log 2>&1 &`).

---

## 4. PHASE 0: Data

### 4.1 Sampling
- Stream each config (`load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)`). Do not download everything.
- Keep an article only if `sha1("keep"+title) % 5 == 0` (deterministic ~20% subsample). Assign it to a split by `sha1("split"+title) % 1000`:
  - buckets 0–849 → `train` (PT: `pt_train`; EN: `en_forget_train`)
  - 850–899 → `val`, 900–949 → `test`, 950–999 → `relearn_fresh` (EN only; PT ignores this bucket)
- Once a split has reached its quota of characters **after cleaning**, skip further articles for it; stop when every split is full:

| Split | Characters |
|---|---|
| pt_train | 80M |
| pt_val, pt_test | 2M each |
| en_forget_train | 80M |
| en_val, en_test | 2M each |
| en_relearn_fresh | 5M |

- **Smoke-test dataset first:** quotas of 2M/0.2M/0.2M. Run the whole pipeline end to end on it before scaling.

**Why (Laura).** Splitting by *article hash* (never by chunk) prevents leakage: chunks of one article would otherwise appear in both train and validation, making held-out scores look better than they are. Strided streaming avoids downloading the whole dump but is not a perfectly random sample. State that as a limitation.

### 4.2 Cleaning
1. Unicode NFC normalization, lowercase.
2. Allowed characters (everything else becomes `UNK`):
   - letters `a-z`
   - accented `á à â ã ç é ê í ó ô õ ú`
   - digits `0-9`
   - space and punctuation `. , ; : ! ? - ( ) ' "`
3. Replace newlines and runs of whitespace by a single space. Documents are joined with an `EOS` symbol.
4. Drop any article whose `UNK` rate is above 0.5%.
5. Vocabulary: ~60 characters + `UNK` + `EOS` (~63). Assert vocab size ≤ 255 (data stored as `uint8`). Save to `data/vocab.json`.

### 4.3 Language purity
Wikipedia articles mix languages. A Portuguese corpus that contains English would break the premise "θ_ref never saw English".
- For each article, compute the share of whitespace tokens in an English stopword set and in a Portuguese stopword set.
- Starting sets (tune by inspection):
  - EN: `the and of to in is was that with for from this which were his her not had has have are by`
  - PT: `de que e os do da dos das em um uma para com não por mais na nas nos foi ao ele ela como mas são seu sua`
  - Ambiguous words (`a, as, no, se, o`) are deliberately excluded.
- Rule for a PT article: drop if `en_share > 0.05` or `en_share > pt_share`. Mirror rule for EN articles.
- Inspect the histogram and **20 random dropped articles per language** and adjust the threshold. Record counts in `stats.json`.

### 4.4 Packing
- Concatenate cleaned articles (with `EOS`) per split into one `uint8` binary file: `data/processed/{split}.bin`.
- Training batches: sample random start offsets, take `block_size + 1` characters (inputs and shifted targets). Seeded sampler.

### 4.5 Word lists (for the leakage metric)
- Extract words with regex `[a-záàâãçéêíóôõú]{3,}` from `pt_train` and `en_forget_train`. Compute frequency per million words in each.
- `EN_only`: `f_en ≥ 2` per million and `f_pt ≤ 0.02 · f_en`. `PT_only`: mirror rule.
- `REAL`: words with `f ≥ 0.5` per million in either corpus (excludes typos and one-offs).
- Save to `data/wordlists.json` with sizes.

### 4.6 Tests (`tests/test_data.py`)
- No article title appears in two splits.
- `en_relearn_fresh` is disjoint from `en_forget_train`.
- Vocabulary round trip (encode then decode equals text).
- UNK rate under threshold in every split.

### STOP gate 0: show Laura
- `stats.json`: characters, articles, dropped counts, UNK rates per split.
- 5 random cleaned snippets per language, and the list sizes of `EN_only`, `PT_only`, `REAL`.
- 10 of the most frequent `EN_only` words (they should be unmistakably English) and 10 of `PT_only`.

---

## 5. PHASE 1: Model (Laura writes `model.py`)

**Spec.**

```
vocab_size ≈ 63, block_size = 512
n_layer = 6, n_head = 4, d_model = 256, dropout = 0.1
learned absolute position embeddings, pre-LayerNorm blocks
attention: causal self-attention (d_head = 64)
MLP: Linear(d, 4d) → GELU → Linear(4d, d)
final LayerNorm → tied output head (weights shared with token embedding)
init: normal(0, 0.02), residual output projections scaled by 1/sqrt(2*n_layer)
```
Parameter count ≈ 6 · 12 · 256² (blocks) + 512·256 (positions) + 63·256 (tokens) ≈ **4.9M**.

**Implementation order for Laura.**
1. Attention by hand (explicit matrix multiply, mask, softmax). Verify it matches `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)` numerically.
2. Switch training to the fast SDPA path.
3. Block, then GPT with a `generate()` method (temperature, top-k).

**Tests (`tests/test_model.py`).**
- Output shape `(B, T, V)` and parameter count within 1% of the formula.
- **Causal mask test:** change a character at position `t`; logits at positions `< t` must not change.
- **Initial loss ≈ ln(V)** (about 4.14 nats ≈ 5.98 bits). A big deviation means a bug.
- **Overfit test:** a single small batch must reach loss near 0 within a few hundred steps.
- Bits per character: `bpc = loss_in_nats / ln(2)`.

### STOP gate 1: show Laura
All tests passing, parameter count, initial loss, the overfit curve, and one (gibberish) sample from the untrained model.

---

## 6. PHASE 2: Stage 1, Portuguese only (θ_ref)

**Config.**

```
batch_size = 64, block_size = 512  → 32,768 characters per step
optimizer = AdamW(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1), grad_clip = 1.0
schedule = linear warmup 200 steps, cosine decay to 1e-4
dropout = 0.1 (raise to 0.2 only if validation bpc rises while train bpc falls)
```

**Choosing `max_steps` (timing run first).**
- Training cost ≈ `6 · N_params · characters_seen` FLOPs. With N ≈ 4.9M, each step costs about 1e12 FLOPs.
- Run 100 steps on the target device, measure steps per second, then set `max_steps` so one Stage-1 run takes roughly 30 to 40 minutes. Record the number and its justification.
- Watch for overfitting: 80M characters seen several times by a 4.9M-parameter model. Log train vs val bpc.

**Logging and checkpoints.**
- Every 500 steps: bpc on 100 fixed batches of `pt_val` (same batches every time) **and** on `en_val` as a diagnostic (expected to be worse).
- Checkpoint every 1000 steps (atomic write, keep last 2, plus best by `pt_val`), including model, optimizer, step, config and RNG state.
- Final model saved as `checkpoints/ref_s{seed}.pt`, then `chmod a-w`. Record its SHA-256 in `results/ref_hashes.json`. **Never overwrite.**

**After training.**
- Compute bpc on `pt_val`, `pt_test` (once), `en_val`.
- Generate 200 samples (protocol in §9.2) and save them.
- Record `θ_ref`'s English bpc: this is **the number forgetting aims at**.

**Seeds.** Seed 0 first. Seeds 1 and 2 only after the full pipeline works on seed 0.

### STOP gate 2: show Laura
Training and validation curves, final bpc (PT and EN), 10 sample generations, the timing-run numbers, and the SHA of the reference.

---

## 7. PHASE 3: Stage 2, adding English

**Primary: θ_b (mixed).** Initialize from `ref_s{seed}.pt`, new optimizer. Each step uses 32 sequences from `pt_train` and 32 from `en_forget_train`. AdamW lr 3e-4 → 3e-5 cosine, warmup 100, `K = 0.3 × stage-1 steps`.

**Diagnostic: θ_en (pure English).** Same init, 64 English sequences per step, 500 steps. Purpose: show how fast Portuguese ability degrades and what generations look like.

**Why (Laura).** Pure-English training mixes two effects: learning English and *catastrophically forgetting* Portuguese. If you then "unlearn" with Portuguese descent, any improvement could come from simply re-learning Portuguese. The mixed setting makes the request realistic ("remove English from a bilingual model") and isolates unlearning. The pure-English run is kept to reproduce the effect you want to show.

**Record for θ_b and θ_en:** `pt_val` bpc, `en_val` bpc, leakage on samples (§9.2).

### STOP gate 3: show Laura
Table of bpc for θ_ref, θ_b, θ_en on PT and EN; samples from each; leakage numbers. Expect: English bpc falls a lot; Portuguese changes little for θ_b and much more for θ_en. If not, discuss before continuing.

---

## 8. PHASE 4: Unlearning methods (applied to θ_b)

**Allowed inputs:** θ_b weights, `en_forget_train`, `pt_train`, `pt_val`, `en_val`. **Forbidden:** θ_ref in any form (see test in §11).

Shared settings: AdamW(weight_decay=0), grad clip 1.0, fresh optimizer, each step uses 32 English + 32 Portuguese sequences, up to 2000 steps, evaluation and a saved checkpoint every 100 steps. Let `L_x(θ)` be mean next-character cross-entropy on batch `x`.

| ID | Method | Loss per step |
|---|---|---|
| M0 | Do nothing | none (θ_b itself) |
| M1 | Retain-only control | `L_PT(θ)` |
| M2 | Gradient ascent (GA) | `−L_EN(θ)` |
| M3 | NegGrad+ | `−L_EN(θ) + λ · L_PT(θ)`, λ = 1 |
| M4 (stretch) | NPO | `(2/β) · mean[ log(1 + exp(β · r)) ]` on English, plus `λ · L_PT(θ)`, with `r` = mean per-character log-ratio `log π_θ − log π_b` of the sequence, β = 0.1 |

Notes:
- NPO uses θ_b (the pre-unlearning model) as its reference policy, which is allowed. Document the per-character normalization choice.
- **Divergence guard:** stop a run if `L_EN` on the training batch exceeds `ln(V) + 1` nats (the model is worse than random on English, a sign of collapse), and save the last good checkpoint.
- **Hyperparameter grid** (seed 0 only, selected using validation data), then **frozen for seeds 1 and 2**: `lr ∈ {2e-5, 5e-5, 1e-4}`, λ ∈ {0.5, 1, 2} for M3.

**Operating-point protocol (to avoid peeking at the reference).** Report full trajectories of every method (EN val bpc vs PT val bpc over steps) and two single points:
- **Practitioner rule:** the last checkpoint before `pt_val` bpc worsens by more than 2% relative to θ_b, or the final step. Uses no reference information.
- **Oracle stopping (best case, label it as such):** the first checkpoint whose `en_val` bpc reaches θ_ref's. This is optimistic because it uses the reference.

**Why (Laura).** Unlearning papers often tune against the retrained model; that gives results better than anyone could get in practice. Reporting both makes your claim honest. GA alone has an unbounded loss, which is why it explodes; adding descent on retained data anchors the model.

### STOP gate 4: show Laura
Trajectory plot for all methods, the two operating points per method, sample generations at the practitioner point, and any divergence events.

---

## 9. PHASE 5: Evaluation

Evaluate: θ_ref, θ_b, θ_en, and θ_u at both operating points for M1, M2, M3 (M4 if run). Test splits are used here for the first time.

### 9.1 Quantitative metrics

| Metric | Definition | Reading |
|---|---|---|
| bpc (PT test, EN test) | mean cross-entropy / ln 2 on 500 fixed chunks | Utility (PT) and forgetting (EN). EN bpc above θ_ref's = over-unlearning |
| bpc on train subsets | same on 500 chunks of `pt_train` and `en_forget_train` | Memorization vs generalization |
| KL to reference | mean per-character `KL(p_ref ‖ p_model)` teacher-forced on 500 chunks of PT test and EN test | Distance to the reference model's behaviour |

### 9.2 Generation and language leakage
- 200 prompts per language: 64-character prefixes taken from `pt_test` / `en_test` (cut at a word boundary). Generate 400 characters, temperature 0.8, top-k 40, seed fixed per prompt.
- Extract words (regex as in §4.5, length ≥ 3).
- **EN-leakage** = (generated `EN_only` words) / (generated `EN_only` + `PT_only` words).
- **Real-word rate** = share of generated words in `REAL`.
- Report both for PT prompts and EN prompts.

### 9.3 Relearning attacks
For θ_ref and each θ_u (practitioner point), fine-tune with AdamW lr 1e-4 for 300 steps, batch 32, and track `en_val` bpc every 25 steps:
- **n = 0 (zero-example attack):** fine-tune on Portuguese only. Does English bpc recover?
- **n ∈ {100k, 1M, 5M} characters of fresh English** (`en_relearn_fresh`), mixed with Portuguese.
- **Seen variant:** the same sizes taken from `en_forget_train`.
- **Dormancy gap** = `bpc_ref(n) − bpc_unlearned(n)` at the end. A gap above 0 means the unlearned model relearns English faster than the reference does, so English was suppressed, not removed.

### 9.4 Representation analysis (stretch, only if time remains)
- Linear CKA at the last layer between each model and θ_ref on 2k fixed English chunks and 2k Portuguese chunks.
- Layer-wise linear probe (mean-pooled hidden states) for language identification (PT vs EN). Compare probe accuracy of θ_u vs θ_ref. Caution: θ_ref can also separate the languages from letter statistics, so read this as a relative comparison only.

### 9.5 Statistics
- 3 seeds → report mean ± std; also show min–max.
- Reference variability band from the three θ_ref models. State clearly that n = 3 is a weak estimate.
- If only one seed is possible: use bootstrap confidence intervals over evaluation chunks and say so in the limitations.

### 9.6 Figures
1. **Trajectory:** x = EN val bpc, y = PT val bpc, one curve per method; mark θ_ref and θ_b.
2. **Relearning curves:** EN val bpc vs steps, θ_ref vs θ_u, for each n.
3. **Table:** bpc, KL, leakage and real-word rate for all models.
4. **Training curves** for stages 1 and 2 (appendix).

### STOP gate 5: show Laura
All tables and figures, plus a plain-language list of what each result means for H1–H3.

---

## 10. Experiment matrix, compute and schedule

**Per seed.** Stage 1 (X minutes) → Stage 2 mixed (~0.3X) → Stage 2 pure-EN (small) → unlearning grid (seed 0 only) / chosen configs (other seeds) → evaluation → relearning (cheap: 300 steps per cell).

**Minimum viable result (MVP, about 8 working hours).** Seed 0 (+ seeds 1 and 2 if time) × {θ_ref, θ_b, M1, M2, M3} × {bpc, KL, leakage, relearning n = 0 and one n > 0}. Everything else is stretch: M4/NPO, representation analysis, mirrored experiment (English base, forget Portuguese), native-Portuguese check.

| When | Work |
|---|---|
| Mon 16 (today) | Phase 0 on the smoke dataset, Phase 1 tests, start timing run |
| Tue 17 | Full data (Gate 0), Stage 1 seed 0 (Gate 2) |
| Thu 19 | Stage 2 (Gate 3), unlearning methods and grid on seed 0 (Gate 4) |
| Fri 20 (optional) | Seeds 1 and 2 stage 1 and 2 |
| Tue 24 | Evaluation and relearning (Gate 5) |
| Thu 26 | Figures, 2-page IEEE report |
| Sat 28 (optional) | Buffer: stretch items, interview rehearsal |

If behind after Thursday 19, cut stretch items first. Protect the report time.

---

## 11. Rules for Claude Code

1. **Stop at every gate.** Show the requested outputs and wait for approval before continuing.
2. **Confirm data at each pipeline stage** before building on it.
3. **Model code is Laura's.** Create `model.py` with class and method signatures, docstrings with tensor shapes, `raise NotImplementedError` bodies, and the failing tests in `tests/test_model.py`. Implement everything else yourself. Only if Laura explicitly says "implement the model", write it, then walk her through it line by line.
4. **Isolation test (`tests/test_isolation.py`).** `unlearn.py` and any file it imports must not reference `ref_s*.pt`, `checkpoints/ref`, or θ_ref in any form; the test greps for it. The reference is loaded only by `evaluate.py` and `relearn.py`.
5. **No tuning on test splits.** Use validation splits for any choice. Touch test splits only in Phase 5.
6. **Do not change the evaluation protocol** (§9) after seeing results. If something must change, record the change and reason in `results/deviations.md`.
7. **Reproducibility.** Seed Python, NumPy and PyTorch; store configs next to every checkpoint; append one JSON line per evaluation to `results/*.jsonl`. Note that GPU/MPS runs may still be nondeterministic.
8. **Fix root causes.** If a test or run fails, find and fix the cause; no workarounds or silent skips.
9. **Explain.** After each phase, write a 10-line plain-language summary of what the code does and why, in `docs/phase_N.md`, for Laura.
10. **Scaling.** Ask before launching any run longer than ~40 minutes or increasing data/model size.
11. **Text I/O** always with explicit UTF-8.
12. **Do not overwrite** `ref_s*.pt`; it is read-only.

---

## 12. For Laura: concepts and interview questions

**Concepts to be able to explain without notes**
- Forget set, retain set, exact vs approximate unlearning, why retraining is the gold standard, and why here θ_ref is the gold standard.
- Why the unlearning method must not use θ_ref.
- Cross-entropy, bits per character, perplexity, and what the initial loss ≈ ln(V) means.
- Causal self-attention and why the mask matters; what "tied embeddings" are.
- Why GA diverges and why adding retain-set descent helps. What NPO changes.
- Why the retain-only control exists (confound).
- What KL(p_ref ‖ p_model) measures and why direction matters.
- What a relearning attack tests and what a positive dormancy gap means.
- Why oracle stopping is optimistic.
- Why splitting by article (not chunk) matters.

**Likely interview questions**
- "Explain your work in 30 seconds."
- "Why a character-level model? What are its limits?"
- "Why is θ_ref the right reference? What would exact retraining look like here?"
- "Why NegGrad+ instead of plain gradient ascent? What did your control show?"
- "Your English and Portuguese share letters. How does that affect forgetting?"
- "How do you know the model forgot the language and did not just change its outputs?"
- "What did you do about English inside the Portuguese Wikipedia?"
- "Which result would have falsified your hypothesis?"
- "How does your work differ from multilingual LLM unlearning papers?"

---

## 13. Report mapping (IEEE, 2 pages including references)

- **Introduction:** why remove a language or data source from a trained model; the research question and its relevance; relation to prior work.
- **Related work / gap (short table):**
  - LLM unlearning with gradient ascent and retain terms (Yao et al. 2024; Yao et al. on pre-trained LLMs).
  - Relearning and tamper resistance (Siddiqui et al.; Deeb & Roger).
  - Representation-level evaluation (Yong et al.).
  - Multilingual unlearning in pretrained LLMs (cross-lingual transfer studies).
  - **Gap:** nothing found on unlearning a whole language from a small from-scratch model with an exact reference. State this as "to our knowledge" and describe how you searched.
- **Methods:** corpus, cleaning, model, stages, methods M1–M3, metrics, relearning protocol, operating points.
- **Results:** Figures 1 and 2, the table, and a short answer for each of H1–H3.
- **Conclusion:** the "so what", limitations (machine-scale, Wikipedia domain differences, 3 seeds, oracle stopping), what you learned.

---

## 14. Risks and open decisions

**Risks**
- Shared letters: unlearning English will also damage Portuguese; the retain term and the control matter.
- Portuguese Wikipedia contains English; the purity filter is imperfect, so describe "close to never saw English".
- Small model: expect correct spelling patterns and weak meaning. Judge by metrics, not fluency.
- The course rubric lists training on a small dataset as a flaw: justify data and model sizes and show validation curves.
- Compute: if the timing run shows Stage 1 is too slow, reduce `max_steps` or model width before reducing seeds.

**Open decisions for Laura**
1. Number of seeds (3 preferred, 1 as fallback).
2. Run NPO (M4) or not.
3. Run the mirrored experiment (English base, forget Portuguese) if time allows.
4. Final deadline: it determines how much of the stretch list is realistic.