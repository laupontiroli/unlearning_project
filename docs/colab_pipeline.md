# Colab pipeline for phases 3–5

> **Scope since 2026-11-24: deviation D5** (one seed, 6 unlearning runs of ≤1000 steps, 2 relearning conditions). Where this file says 3 seeds, a 15-run grid or 7 relearning conditions, D5 overrides it. Code: `unlearn.py`, `points.py`, `relearn.py`, `scripts/phase5_eval.py`.

Draft, 2026-11-24. Nothing in Phases 3–5 runs before its gate is approved. This file only fixes *where* and *how* things run.

## Why Drive becomes the working copy

The unlearning grid alone has 15 runs on seed 0, each with up to 20 checkpoints of about 20 MB. That is far too much to download to the Mac. So from Gate 2 on:

- `MyDrive/unlearn/checkpoints/` and `MyDrive/unlearn/results/` are the **working copy**. Every notebook links `checkpoints/` and `results/` to them, so all scripts run unchanged.
- The Mac keeps the code, the docs, `results/deviations.md`, and a **gate pack** for each gate. A gate pack is `gate_packs/gateN_results.zip`: all of `results/`, without checkpoints, a few MB. Laura unzips it into `colab_results/gateN/` and Claude writes the gate report from it.
- θ_ref and θ_b are also downloaded once each (20 MB) as backups. On the Mac they are made read-only.

## Files on Drive

```
MyDrive/unlearn/
  unlearn_bundle.tar.gz   data + old code (60 MB, uploaded once, never rebuilt)
  code_bundle.tar.gz      latest code (0.4 MB, re-upload after every code change)
  checkpoints/            ref_long_s0.pt, theta_b_s0.pt, theta_en_s0.pt, unlearn/..., relearn/...
  results/                train_*.jsonl, eval.jsonl, test_access.jsonl, *_hashes.json, samples/, figures/
  gate_packs/             gate2_results.zip, gate3_results.zip, ...
```

`uv run python scripts/make_colab.py` rebuilds `colab/code_bundle.tar.gz` and the notebooks. The code bundle also carries a snapshot of the Mac `results/` as `results_mac/`. Notebook cell 5 merges that snapshot into Drive and **never overwrites** anything there: missing files are copied, hash records are merged key by key (a conflicting SHA stops the merge), and JSONL logs get only the missing lines. This tool is `scripts/colab_sync.py` and is covered by `tests/test_colab.py`.

## Protecting θ_ref without chmod

Google Drive does not support `chmod`, so the SHA-256 record does the protecting:

- `colab_sync.py verify` re-hashes every checkpoint listed in `ref_hashes.json` and `model_hashes.json`.
- It runs at the **start and end of every notebook** and raises an error if any file changed.
- The isolation rule is unchanged: `unlearn.py` and its imports never name θ_ref (`tests/test_isolation.py`, Phase 4).

## Shared setup (cells 1–5 of every notebook)

1. Set `SEED` and check the GPU.
2. Mount Drive.
3. Unpack the data bundle, then the code bundle on top. Run `sha256sum -c` on the data and link `checkpoints/` and `results/` to Drive.
4. Install `pyyaml` and run `pytest`.
5. Merge the Mac results into Drive, then run `verify`.

Every later cell can be re-run after a disconnect. Training cells resume from the last checkpoint, and work already done is skipped. Test-split cells are skipped once they have run, because a second test access is refused.

## Notebooks

| Notebook | Phase / gate | Status | Runs | Gate pack holds |
|---|---|---|---|---|
| `stage1_long.ipynb` | Phase 2, Gate 2 | running | θ_ref: 6000 steps | (downloaded by hand) |
| `02_gate2_eval.ipynb` | Gate 2 redo | **ready** | Curves; bpc on pt_val, en_val and train splits; **pt_test once**; 200 + 200 generations from val prompts | `gate2_results.zip` |
| `03_stage2.ipynb` | Phase 3, Gate 3 | **ready (draft configs)** | θ_b: 1800 mixed steps; θ_en: 500 EN steps; both evaluated on val + train splits | `gate3_results.zip` |
| `04_unlearn.ipynb` | Phase 4, Gate 4 | **ready** (D5 scope) | 6-run grid, operating points, trajectory figure, practitioner samples | `gate4_results.zip` |
| `05_eval_relearn.ipynb` | Phase 5, Gate 5 | **ready** (D5 scope) | Test-split evaluation (bpc ± 95% CI, KL, leakage), relearning n0 / fresh1M, table, figure | `gate5_results.zip` |

Stage 2 configs: [configs/stage2_mixed.yaml](../configs/stage2_mixed.yaml) and [configs/stage2_en.yaml](../configs/stage2_en.yaml). They follow PLAN §7:
- init from `ref_long_s{seed}.pt` (D4), with a fresh optimizer;
- lr 3e-4 → 3e-5, warmup 100;
- weight decay 0.1, as in Stage 1;
- the evaluation uses the same 100 fixed batches × 16 sequences as Stage 1, so all bpc curves are comparable.

Both configs pass a 3-step CPU smoke run.

### 04_unlearn.ipynb (plan only, code not written)

- **New code:**
  - `src/unlearnlang/unlearn.py`: methods M1–M3 (and M4 if chosen) and the divergence guard (`L_EN > ln 62 + 1 = 5.13` nats).
  - `scripts/unlearn.py`
  - `configs/unlearn_*.yaml`
  - `tests/test_isolation.py`
- **Inputs:** only `theta_b_s{seed}.pt` and the allowed splits. Each step uses 32 EN + 32 PT sequences, AdamW with weight_decay 0, and a fresh optimizer. Every 100 steps it evaluates pt_val and en_val, using the same fixed batches as Stage 1 and 2.
- **Grid, seed 0, 15 runs:**

  | Method | Settings | Runs |
  |---|---|---|
  | M1 | lr ∈ {2e-5, 5e-5, 1e-4} | 3 |
  | M2 | lr ∈ {2e-5, 5e-5, 1e-4} | 3 |
  | M3 | lr ∈ {2e-5, 5e-5, 1e-4} × λ ∈ {0.5, 1, 2} | 9 |

  Each run has at most 2000 steps. One run is the unit of resumption: an unfinished run restarts from scratch, with the same seed.
- **Storage:**
  - Grid runs keep their full trajectory (JSONL) and only the final weights.
  - The selected configuration of each method is then re-run, keeping weights every 100 steps. The operating points need those checkpoints.
  - This costs about 3 × 2000 extra steps, but saves roughly 5 GB of Drive space.
- **Operating points** come from a separate `scripts/select_points.py`, which is evaluation-side code:
  - The *practitioner* point uses only θ_b's pt_val.
  - The *oracle* point uses θ_ref's en_val, read from the final eval line of `train_stage1_long_s0.jsonl`. That line used the same fixed batches, so no θ_ref weights are loaded during unlearning.
- **Seeds 1 and 2** (if chosen) run Stage 1 long, then Stage 2, then the three frozen configurations.

### 05_eval_relearn.ipynb (plan only)

- **Evaluation on the test splits, once each:** θ_ref, θ_b, θ_en, and θ_u at both operating points for each method. It reports bpc, KL to θ_ref (`kl_to_reference` already exists; `eval_checkpoint.py` needs a `--kl-ref` option), leakage and real-word rate from test prompts.
- **Relearning** (`relearn.py`, which may load θ_ref):
  - Models: θ_ref and θ_u at the practitioner point.
  - Seven data conditions: n = 0; fresh English with n ∈ {100k, 1M, 5M} characters; and the "seen" variant with the same three sizes.
  - Each run is 300 steps at batch 32, with en_val measured every 25 steps.
  - On seed 0 that makes 4 models × 7 conditions = 28 runs, or 8,400 steps.
- Figures as in PLAN §9.6.

## Compute budget (filled in from the T4 timing run)

With r = steps/s on the T4, from `results/timing.jsonl` on Drive:

| Block (seed 0) | Steps |
|---|---|
| Stage 2 (θ_b + θ_en) | 2,300 |
| Unlearning grid (≤) | 30,000 |
| Re-run of the 3 selected configs | 6,000 |
| Relearning | 8,400 |
| **Total** | **≈ 47,000 steps + evaluations** |

Each extra seed adds Stage 1 (6,000), Stage 2 (2,300), the 3 configs (6,000) and relearning (8,400), about 23,000 steps. Free Colab sessions last about 12 h at most and have a daily GPU quota, so each notebook is built to be stopped and resumed.

**Measured 2026-11-24 (stage1_long seed 0, T4):** 0.704 s/step at 64 × 512 characters. At that speed:
- seed 0 after Stage 1 (≈ 47k steps) takes ≈ 9.2 h;
- each extra seed (≈ 23k steps) takes ≈ 4.5 h;
- 3 seeds take ≈ 18 h of T4 time in total, plus evaluations.

A rough FLOP count gives only ≈ 3 TFLOP/s, well below what a T4 can do in fp16. Check the throughput before committing that much GPU time.

## Decisions still open (for Laura, at the gates)

1. ~~Seeds~~ **Decided 2026-11-24: 3 seeds (0, 1, 2).** Seeds 1 and 2 each need their own Stage 1 long run (6,000 steps) before their Stage 2; the grid is still tuned on seed 0 only and frozen for seeds 1 and 2.
2. **NPO (M4)**: yes or no. It adds 3 grid runs and 1 selected config.
3. **Grid selection rule** (Gate 4): PLAN §8 says "selected using validation data" but gives no rule. A proposal, to fix *before* the grid runs: for each method, the configuration with the lowest pt_val bpc at its practitioner point among those whose en_val rises by at least X bpc. It must not use θ_ref.
