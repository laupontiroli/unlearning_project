# Phase 4: unlearning English from θ_b (Gate 4)

1. Methods, all starting from θ_b with 32 EN + 32 PT sequences per step, AdamW without weight decay, constant lr, at most 1000 steps (D5):
   - M1, Portuguese descent only (control);
   - M2, gradient ascent on English;
   - M3, NegGrad+ = −L_EN + λ·L_PT.

   The code never loads θ_ref (`tests/test_isolation.py`).
2. First grid: M2 and every M3 run crossed "worse than uniform on English" (the guard, 7.4 bpc) within 21–66 steps, before the first evaluation at step 100. D6 added an evaluation and checkpoint every 5 steps for the first 100 steps and reran M2/M3 unchanged (`configs/unlearn_v2.yaml`).
3. Validation bpc, from the same fixed batches as Stages 1–2. θ_ref: PT 1.619, EN 2.534. θ_b: PT 1.635, EN 1.852. The EN gap to close is 0.68 bpc. Practitioner rule: PT ≤ 1.02 × 1.635 = 1.667.

   | Run | Guard hit at step | Practitioner point: step, PT / EN (% of EN gap closed) | Oracle point: step, PT / EN |
   |---|---|---|---|
   | M1 lr 5e-5 | never | 1000: 1.617 / 1.946 (14%) | never reached |
   | M2 lr 5e-5 | 21 | 5: 1.644 / 1.893 (6%) | 15: 1.985 / 4.145 |
   | M3 lr 2e-5 λ 1 | 53 | 20: 1.650 / 1.989 (20%) | 35: 1.764 / 3.064 |
   | M3 lr 2e-5 λ 2 | 66 | 30: 1.656 / 2.137 (42%) | 40: **1.704** / 2.757 |
   | M3 lr 5e-5 λ 1 | 26 | 10: 1.656 / 2.013 (24%) | 15: 1.720 / **2.625** |
   | M3 lr 5e-5 λ 2 | 33 | 15: 1.665 / **2.215 (53%)** | 20: 1.724 / 2.966 |

4. **No run passes through θ_ref.** When EN reaches θ_ref's 2.53, PT is already 1.70–1.77, i.e. +0.09 to +0.15 bpc worse than θ_ref. The two languages share letters and subwords, so ascent on English also damages Portuguese.
5. **The reference-free stopping rule stops early.** At the practitioner point, at most 53% of the EN gap is closed (M3, lr 5e-5, λ 2). Reaching θ_ref's EN needs the oracle, i.e. the reference.
6. **H1: supported.** M2 is uncontrollable. Within 5 steps it jumps from EN ≈ 2.3 to 4.1, at a PT cost of +0.37. Its first checkpoint past the target is far beyond it, and it passes uniform soon after.
7. **M3 is controllable for a short window.** A larger λ costs less PT for the same EN gain. Every M3 run still collapses past uniform if it is not stopped (step 26–66): the retain term delays the collapse of the unbounded ascent term but does not prevent it.
8. **H3, on bpc: the control forgets less.** M1 closes only 14% of the EN gap in 1000 steps while PT improves to 1.617, below θ_b. Generation tells a different story (validation prompts): M1 lowers EN leakage on EN prompts from 0.95 to 0.33 and on PT prompts from 0.029 to 0.005. It hides English in sampling without removing it from the predictions.
9. **Selected for Phase 5 (D5 rule: highest EN val at the practitioner point): M3 lr 5e-5 λ 2** (practitioner = step 15).
10. Practical note: all the action happens in the first 15–40 steps. Even 5-step resolution is coarse, because EN moves up to 0.3 bpc between checkpoints.
11. Generation at the practitioner points: 200 validation prompts per language, 500-chunk bpc. In the notebook output each `bpc:` line comes *before* the `====` header of its run. The mapping below was checked against the training-time bpc.

    | Model | PT val | EN val | EN-leak, PT prompts | EN-leak, EN prompts | Real-word rate, PT prompts |
    |---|---|---|---|---|---|
    | θ_ref | 1.620 | 2.527 | 0.001 | 0.075 | 0.94 |
    | θ_b | 1.638 | 1.845 | 0.029 | 0.954 | 0.94 |
    | M1 (step 1000) | 1.620 | 1.938 | 0.005 | 0.326 | 0.94 |
    | M2 (step 5) | 1.646 | 1.887 | 0.027 | 0.925 | 0.92 |
    | M3 2e-5 λ1 (step 20) | 1.652 | 1.980 | 0.002 | 0.148 | 0.91 |
    | M3 2e-5 λ2 (step 30) | 1.658 | 2.126 | 0.002 | 0.059 | 0.92 |
    | M3 5e-5 λ1 (step 10) | 1.658 | 2.005 | 0.002 | 0.103 | 0.91 |
    | M3 5e-5 λ2 (step 15) | 1.666 | 2.205 | 0.003 | 0.044 | 0.92 |

12. **Leakage overstates forgetting.**
    - Every M3 practitioner point already looks like θ_ref in generation: it rarely writes English (EN-prompt leakage 0.04–0.15, against θ_ref's 0.075), yet bpc shows that only 19–53% of the English gap is closed.
    - The model still predicts English well when forced to read it; it just stops *producing* it.
    - The bpc / leakage gap is a central result.
13. **Visible side effects in the samples:**
    - Repetition loops on EN prompts ("ohu ohu ohu…", "cantores da radiação cantores da radiação…").
    - Invented PT words ("guilómetros", "negocietal").
    - Real-word rate on PT prompts falls from 0.94 to 0.91–0.92.
    - Some continuations start in English, then switch mid-sentence to Portuguese ("rock from a zootrophic variedade da medicina…").
    - M2 at step 5 is essentially θ_b: it still writes the English Wikipedia footer ("references external links 1926 births…").
- Gate 4 approved by Laura on 2026-11-25 ("Approve, with oracle"); D7 adds relearning from the M3 oracle point.
