# Deviations from PLAN.md

## D1 (2026-11-16): Phase 0 sampling: random row groups instead of a prefix of the stream (§4.1)

**Plan.** Stream each dump from the start, keep articles with `sha1("keep"+title) % 5 == 0`, and stop when every split is full.

**Problem (seen on the smoke run).** The dumps are ordered by article id, so the stream starts with the oldest, longest and best-known articles (smoke articles averaged about 20k characters). At full scale this would have read about 20% of the Portuguese dump but only about 2% of the English one, so the two corpora would differ in more than language. Validation and test (5% buckets) would also fill from an even earlier part of the dump than train.

**Change.** Each dump's parquet files are split into row groups of about 1000 consecutive articles: 1116 row groups for PT and 6437 for EN. These row groups are visited in a seeded random order (`sampling_seed` in the config). Inside each row group the plan's rules are unchanged: the keep-hash (`% 5`), the split hash, cleaning, the UNK and purity filters, and the quotas. Every split therefore fills uniformly across the whole dump, and only the row groups actually used are downloaded.

**Unchanged.** Splits are still assigned by title hash (no leakage), and cleaning, filters and quotas are as in the plan.

**Remaining limitation.** This is cluster sampling: articles inside a row group were created around the same time, which is mild clustering, not a perfectly random sample. `stats.json` records `row_groups_per_dump_decile` to show the coverage.

**Approved by** Laura at Gate 0 ("Yes, with stride fix").

## D2 (2026-11-16): typography normalization (§4.2)

Curly quotes, «», en and em dashes, the minus sign and non-breaking spaces are mapped to their ASCII equivalents (`' " -` and space) before non-allowed characters are turned into UNK. Config flag: `normalize_typography`.

## D3 (2026-11-16): EN purity filter keeps only the "other language dominates" rule (§4.3)

**Plan.** A mirror rule for EN articles: drop if `pt_share > 0.05` or `pt_share > en_share`.

**Problem (seen at Gate 0).** Of the 119 EN articles dropped, most were genuine English articles about Brazilian, Portuguese or Spanish places and people ("Santa Cecília do Sul is a municipality in…"). They tripped the 5% threshold because of "de", "do" and "da" in proper names. Dropping them biased the English set against Lusophone and Hispanic topics. The experiment's premise only requires the *Portuguese* side to be free of English.

**Change.** `max_other_share` is now set per language: `pt: 0.05` (unchanged) and `en: null` (no absolute threshold). EN articles are still dropped when PT stopwords outnumber EN ones. EN purity drops fell from 119 to 26. The remaining drops are mostly lists of Spanish or Portuguese names and titles. PT data is unchanged.

**Approved by** Laura at Gate 0 ("Relax EN rule only"). The PT threshold stays at 5%. The remaining English in PT (mostly titles and bibliographies: "the" ≈ 1,440/M words but "was" ≈ 8/M) is stated as a limitation.

## D4 (2026-11-24): θ_ref is a longer Stage-1 run on a Colab GPU; the Mac run is kept as a pilot (§6)

**Plan.** Set `max_steps` from a Mac timing run so Stage 1 takes 30–40 minutes, and save it as `checkpoints/ref_s{seed}.pt`.

**Problem (seen at Gate 2).** The Mac run (`ref_s0.pt`, 2000 steps, 0.82 epochs) was still improving at its last step: pt_val went from 1.947 to 1.871 bpc over the final 500 steps. Stage 2 adds 30% more steps that include Portuguese, so θ_b would improve on Portuguese just from the extra training. That makes "return to θ_ref" hard to interpret.

**Change.**
- θ_ref for all later phases is `checkpoints/ref_long_s{seed}.pt`, from `configs/stage1_long.yaml`.
- It uses the same model, data and optimizer, with `max_steps = 6000` (197M characters ≈ 2.5 epochs) and a checkpoint every 500 steps. It is trained on a Colab GPU with `colab/stage1_long.ipynb`.
- Its SHA-256 goes into `results/ref_hashes.json` next to the pilot's.
- `ref_s0.pt` is not modified. It stays read-only and is reported only as a pilot (pt_val 1.874, en_val 2.831, pt_test 1.876, used once).
- Stage 2 length becomes K = 0.3 × 6000 = 1800 steps.

**Approved by** Laura at Gate 2 ("Train longer on Colab").

## D5 (2026-11-24): one-day scope (PLAN §8, §9.3, §9.5, §10)

**Reason.** The project is due 2026-11-25. On the Colab T4 (1.55 steps/s, measured), the full plan needs about 18 h of GPU time. The reduced scope below is close to the plan's own minimum version (§10) and needs about 2.5 h.

**Changes.**
- **Seeds:** 1 (seed 0) instead of 3. Uncertainty is reported as 95% bootstrap CIs over the 500 evaluation chunks (the §9.5 fallback). Per-chunk values are saved in `results/chunks/` for paired comparisons. The single seed is listed as a limitation.
- **Unlearning (§8), focused on NegGrad+ (M3), the project's main method.** 6 runs on seed 0, each up to **1000** steps instead of 2000. Evaluation and checkpoints stay every 100 steps.

  | Method | lr | λ | Runs |
  |---|---|---|---|
  | M1 (retain-only control) | 5e-5 | – | 1 |
  | M2 (gradient ascent) | 5e-5 | – | 1 |
  | M3 (NegGrad+) | 2e-5, 5e-5 | 1, 2 | 4 |

  The roles of the two other methods are fixed in advance. M1 only has to keep producing Portuguese. M2 has to change the model enough to show what unlearning without a retain term does.
- **No M4 (NPO).**
- **Choosing the M3 setting used for relearning, fixed before the grid runs.** Among the 4 M3 runs, pick the one with the highest en_val bpc at its practitioner point. This uses validation data only, no θ_ref. All 4 M3 runs are still evaluated on the test splits.
- **Unspecified details, fixed here.** Unlearning uses a constant lr with no warmup. Relearning uses AdamW with weight_decay 0 and a constant lr of 1e-4. "Mixed with Portuguese" means 16 PT + 16 EN sequences per batch.
- **Relearning (§9.3):** 2 conditions instead of 7: n = 0 (Portuguese only) and n = 1M characters of fresh English. Models: θ_ref, and M1, M2 and the selected M3 at their practitioner points. Each run is 300 steps, with en_val every 25 steps.
- **Not run:** seeds 1–2, the "seen" relearning variant, n ∈ {100k, 5M}, and §9.4 (representation analysis).

**Approved by** Laura on 2026-11-24 ("We can focus on the unlearning since is the focused of the project…").

## D6 (2026-11-25): dense evaluation during the first 100 unlearning steps (PLAN §8)

**Seen in grid v1.**
- M2 (GA, lr 5e-5) hit the divergence guard (English train loss above ln 62 + 1 = 5.13 nats, i.e. 7.4 bpc, worse than uniform) at step **21**.
- M3 (NegGrad+, lr 2e-5, λ = 1) hit it at step **53**.
- Both runs ended before the first evaluation at step 100. Their trajectories therefore have only θ_b and the last good checkpoint, and the operating points cannot be located. In particular we cannot tell whether English passed θ_ref's level (2.53 bpc) while Portuguese was still intact.
- M1 (control, lr 5e-5) is smooth over 1000 steps: PT val 1.635 → 1.617, EN val 1.852 → 1.946.

**Change.** Evaluate and save a checkpoint every 5 steps during the first 100 steps (`configs/unlearn_v2.yaml`, `dense_until: 100`, `dense_interval: 5`); after that, every 100 steps as before. M2 and the 4 M3 runs are rerun with tag `v2`. The method, data, lr, λ, the guard and the metrics are unchanged, and so is the same-seed data stream. M1 is reused from v1. The v1 logs stay on Drive as a record. Operating points come from the v2 grid only (`select_points.py --config configs/unlearn_v2.yaml`).

**Why this is not tuning on results.** It changes only how often the same trajectory is measured. It does not change what is optimized, and no test data is involved.

**Finding to report.** Even with the retain term (λ = 1), NegGrad+ at lr 2e-5 collapses within about 50 steps. This is a result in itself: the unbounded ascent term dominates.

**Approved by** Laura on 2026-11-25 (she chose to run the v2 notebook after the explanation).

## D7 (2026-11-25): extra relearning run from the M3 oracle point (PLAN §9.3)

**Plan.** Relearning starts from θ_ref and from the practitioner point of each method.

**Problem (seen at Gate 4).** The practitioner points keep most of their English. M3 closes 53% of the EN gap, M2 6%, M1 14%. A faster recovery than θ_ref's would therefore be confounded with where each model starts.

**Change.** Also relearn from the oracle point of the selected M3 run (lr 5e-5, λ 2, step 20, EN val 2.97, past θ_ref's 2.53). It is labelled `M3_oracle` and uses the same two conditions and the same protocol. It is reported as an *addition*, and the oracle remains labelled as reference-informed.

**Approved by** Laura at Gate 4 ("Approve, with oracle").

## D8 (2026-11-25): unlearning with proxy English the model never saw (new experiment)

**Motivation.** The report is reframed around unlearning a *concept* (here a language) when the original training data is not available. In the main grid, the forget set is `en_forget_train`, the same pool θ_b was trained on: θ_b's batches covered ≈31% of it. That does not test the "unknown source" case.

**Change.**
- One extra run: the selected M3 configuration (lr 5e-5, λ 2, dense evaluation) with tag `proxy`.
- Its English batches come from `en_relearn_fresh` characters 1M–5M. No model has trained on these: relearning uses only the first 1M, and unlearn.py refuses an offset below 1M.
- Everything else is unchanged: init θ_b, Portuguese retain batches, seed, guard, and operating-point rules (`configs/unlearn_proxy.yaml`).
- Its points go in a separate file, `results/operating_points_proxy_s0.json`, so the Gate 4 selection is not changed.
- Test evaluation: practitioner and oracle points, read once each.
- Relearning: n0 and fresh1M, practitioner and oracle points (labels `M3_proxy`, `M3_proxy_oracle`). θ_ref's relearning runs are reused.

**Not tuning.** The configuration was frozen at Gate 4. Only the source of the forget batches changes.

**Approved by** Laura on 2026-11-25 ("Reframe + run D8").
