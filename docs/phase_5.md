# Phase 5: test evaluation and relearning (seed 0, scope D5)

Test splits were read once per model (`test_access.jsonl`). Fraction of the EN gap closed = (EN_u − EN_b) / (EN_ref − EN_b), with EN test bpc θ_b 1.808 and θ_ref 2.480 (gap 0.672).

1. **Practitioner points (test).** M3 lr5e-5 λ2 (selected): PT 1.669, EN 2.168, so 54% of the gap closed at +0.032 PT bpc over θ_b. M1: PT 1.618, EN 1.901 (14%). M2: PT 1.645, EN 1.850 (6%). The other M3 runs closed 20–42%. No practitioner point reaches θ_ref.
2. **Oracle points (test).** M2 step 15: EN 4.137, overshooting θ_ref by 1.66 bpc; PT 1.983 (+0.36 vs θ_ref); real-word rate 0.69. M3 step 20: EN 2.913 (overshoot +0.43), PT 1.729 (+0.11 vs θ_ref, outside the ±0.026 CI).
3. **KL to θ_ref disagrees with bpc.** On EN, M1 is the closest model to θ_ref (KL 0.44 bits/char), ahead of every M3 (0.52–0.61). M3 oracle has KL_EN 0.89, *higher* than θ_b (0.82): it forgets English in a different way than θ_ref never knew it. On PT, M1 is also closest (0.037, below θ_b's 0.054); every GA-type method moves PT further from θ_ref (0.061–0.127).
4. **Leakage overstates forgetting.** M3 λ2 drops EN leakage on EN prompts from 0.975 to 0.030 (below θ_ref's 0.064) while closing only 54% of the bpc gap.
5. **Relearning, PT + 1M fresh EN** (FixedEval en_val): every unlearned model is at 1.86–1.90 after 25 steps (θ_b: 1.85), including M3 oracle, which starts at 2.97. θ_ref: 2.30 at step 25, 2.09 at step 300. Dormancy gap for M3: +0.44 at step 25, +0.23 at step 300.
6. **Relearning, PT only (n0).** θ_ref stays at 2.53. Unlearned models regain English with *no* English data: M3 goes from 2.22 to 1.92 and M3 oracle from 2.97 to 2.00, both within 25 steps. Dormancy gap for M3 at step 300: +0.59. Only M1 drifts up (1.95 → 1.99).
7. **H1 supported.** GA has a usable window of ~5 steps; at the point where EN reaches θ_ref, PT has degraded by +0.36 bpc and the model collapses 6 steps later.
8. **H2 partly supported.** NegGrad+ does not match θ_ref without paying in PT (practitioner 54% / oracle +0.11 PT). The dormancy claim is strongly supported: English returns within 25 steps, even from Portuguese-only fine-tuning.
9. **H3 supported in bpc, reversed in KL.** M1 raises EN bpc less than M3 (+0.09 vs +0.36), but is distributionally closer to θ_ref on both languages.
10. Paired bootstrap (500 test chunks, vs θ_b), PT bpc: M3 prac +0.032 [0.029, 0.035]; M3 oracle +0.092 [0.085, 0.100]; M2 prac +0.008 [0.008, 0.009]; M1 −0.019 [−0.020, −0.017]. θ_ref has no per-chunk PT test file, because pt_test was read at Gate 2, before chunks were saved. So comparisons with θ_ref on PT stay unpaired. Caveat: 1 seed. Data: `colab_results/gate5/`.

**Gate 5 approved** by Laura on 2026-11-25 ("Approve, fill report").

## D8: proxy unlearning (gate 6 pack, colab_results/gate6)
- M3 lr5e-5 λ2 with English never seen by any model (en_relearn_fresh chars 1M–5M). Collapses at step 30 (original: 33).
- Same validation trade-off curve: at θ_ref's EN level (2.534), PT is 1.692 for proxy vs 1.690 for the original.
- Test: the practitioner point is step 10 (PT 1.651, EN 1.952; paired PT +0.015 [0.013, 0.016] vs θ_b) and the oracle point is step 20 (PT 1.773, EN 3.330, KL_EN 1.30). Both differ from the original only because the 5-step checkpoints land at different places on the same curve.
- Relearning matches the original M3. With n0: practitioner 2.000 → 1.896 and oracle 3.387 → 2.037 at step 25; dormancy gap at step 300 +0.60. With fresh1M: 1.86–1.88 at step 25.
- H4 is supported: unlearning acts on the concept, not on the documents.
