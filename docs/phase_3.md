# Phase 3: Stage 2, adding English → θ_b (mixed) and θ_en (English only)

1. Both runs start from θ_ref = `ref_long_s0.pt` with a fresh AdamW optimizer: lr 3e-4 → 3e-5 with cosine decay, warmup 100 (`configs/stage2_*.yaml`). Trained on a Colab T4 with `colab/03_stage2.ipynb`.
2. θ_b: 1800 steps of 32 PT + 32 EN sequences each (K = 0.3 × 6000). θ_en: 500 steps of 64 EN sequences. SHA-256: θ_b `811e8459de96…`, θ_en `9537938fadfa…` (in `results/model_hashes.json` on Drive).
3. Validation bpc on 500 fixed chunks (95% bootstrap CI over chunks):

   | Model | PT val | EN val |
   |---|---|---|
   | θ_ref | 1.620 | 2.527 |
   | θ_b | 1.638 [1.615, 1.659] | **1.845** [1.821, 1.871] |
   | θ_en | **2.344** [2.322, 2.365] | 1.871 |

4. θ_b learned English (−0.68 bpc) and lost almost no Portuguese (+0.018 bpc). θ_en learned English about as well and lost +0.72 bpc of Portuguese, which is catastrophic forgetting. This matches PLAN §7's expectation.
5. During θ_b training, PT bpc peaks at 1.66 around step 200, then recovers as the lr decays. No overfitting: train and val agree within 0.01 for both runs and both languages.
6. Generation from 200 validation prompts per language. EN-leakage = EN-only / (EN-only + PT-only) generated words.

   | Model | Leakage, PT prompts | Leakage, EN prompts | Real-word rate, PT prompts |
   |---|---|---|---|
   | θ_ref | 0.001 | 0.075 | 0.94 |
   | θ_b | **0.029** (201 EN-only words) | 0.954 | 0.94 |
   | θ_en | **0.952** | 0.992 | 0.91 |

7. θ_b is a real bilingual model: it continues English prompts in English and Portuguese prompts in Portuguese. Some English leaks into its Portuguese, 25× more than in θ_ref.
8. θ_en continues even *Portuguese* prompts in English. This is the "text stops making sense" effect the diagnostic was meant to show.
9. Unlearning targets for Phase 4:
   - raise θ_b's EN val bpc from 1.845 to θ_ref's 2.527 (+0.68);
   - keep PT val near 1.62–1.64;
   - bring PT-prompt leakage from 0.029 back towards 0.001.
10. Colab note: the notebook's test cell showed 7 failures in `tests/test_data.py`. That file needs `*_titles.txt` files, which are not in the bundle; the `.bin` files are SHA-verified in cell 3. The test cell now leaves that file out and says why.
