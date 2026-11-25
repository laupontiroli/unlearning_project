# Phase 2: Stage 1, Portuguese only → θ_ref (plain-language summary)

1. `src/unlearnlang/train.py` is one training loop used by every stage. Each step it takes a batch, computes the cross-entropy loss, backpropagates, clips the gradient norm to 1 and takes an AdamW step. The learning rate warms up linearly, then follows a cosine decay.
2. Weight decay (0.1) applies only to the weight matrices, not to biases or LayerNorm. This is the standard GPT recipe.
3. Every 500 steps the model is scored on the **same** 100 fixed batches of `pt_val`, `en_val` and `pt_train`, with dropout off, so the numbers can be compared over time. If train and val diverge, the model is memorizing.
4. Checkpoints are written atomically (to a temporary file, then renamed). They contain the model, optimizer, sampler and every RNG state, so `--resume` continues **bit-exactly**; `tests/test_train.py` checks this. This was needed: the run was interrupted at step 1000 and resumed.
5. The final model `checkpoints/ref_s0.pt` is made read-only (`chmod a-w`), and its SHA-256 is stored in `results/ref_hashes.json`. The code refuses to overwrite it.
6. Timing run: 0.917 steps/s on the M4 → `max_steps = 2000` (65.5M characters ≈ 0.82 epochs, ~1.9e15 FLOPs) to fit the plan's 30–40 minutes. In practice the second half ran slower (1.56 s/step instead of 1.09), probably because the fanless MacBook Air throttles under sustained heat, so the run took about 47 minutes.
7. Result: pt_val bpc fell from 6.06 (uniform) to **1.87**, beating the order-4 n-gram (2.56). en_val reached **2.84**: θ_ref has never been trained on English text, but it still does far better than chance because of shared letters and words.
8. No overfitting: train bpc = val bpc within 0.002 at every eval. The model has seen less than one epoch.
9. The curve is still going down at step 2000 (−0.08 bpc over the last 500 steps), so θ_ref is **under-trained**. This is a choice for Laura at Gate 2.
10. Evaluation (`src/unlearnlang/evaluate.py`, `scripts/eval_checkpoint.py`) computes bpc on 500 fixed chunks and 200 generations per language. Each prompt has its own seed, so a sample does not depend on which other prompts it was batched with (tested). The test split is used once, and every access is logged in `results/test_access.jsonl`.

## Gate 2 redo: θ_ref = ref_long_s0 (Colab T4, 6000 steps, D4)

11. Evaluated on Colab (`02_gate2_eval.ipynb`, 500 fixed chunks, 95% bootstrap CI over chunks).

    | Split | pilot ref_s0 (2000 steps) | θ_ref ref_long_s0 (6000 steps) |
    |---|---|---|
    | pt_val | 1.874 | **1.620** [1.596, 1.642] |
    | pt_test (read once) | 1.876 | **1.620** [1.597, 1.643] |
    | en_val | 2.831 | **2.527** [2.501, 2.554] |
    | pt_train | 1.880 | 1.611 |
    | en_forget_train | 2.814 | 2.513 |

    SHA-256 `03773ce4d1f8…c5ec646`.
12. No overfitting after 2.5 epochs: pt_train and pt_val differ by 0.009 bpc, which is inside the CI. Training ran at 1.55 steps/s on the T4, and one full evaluation took 200 s.
13. Generation from 200 validation prompts per language:

    | Prompts | EN-leakage | Real-word rate | Words |
    |---|---|---|---|
    | Portuguese | 0.0012 | 0.941 (pilot 0.886) | 8 EN-only vs 6,910 PT-only |
    | English | 0.075 | 0.89 | 409 EN-only vs 5,055 PT-only |

    θ_ref answers English prompts mostly in Portuguese, copying a few English words from the context.
14. English stays 0.91 bpc worse than Portuguese. This is the gap Stage 2 should close and unlearning should reopen.
- Gate 2 approved by Laura on 2026-11-24 ("Approve, run Stage 2").
