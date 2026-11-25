# Unlearning project

The full research plan, with phases, STOP gates and rules, is in `docs/PLAN.md`. Follow its §11 rules. The most important ones:
- Stop at every gate and wait for Laura's approval.
- `model.py` is Laura's to write. Do not implement it unless she explicitly says "implement the model".
- Unlearning code must never touch θ_ref (`checkpoints/ref_s*.pt`, `checkpoints/ref_long_s*.pt`).
- No tuning on test splits.
- All text I/O is UTF-8.

Commands:
- Data: `HF_HOME=$PWD/data/raw/hf uv run python scripts/prepare_data.py --config configs/data_smoke.yaml` (smoke) or `configs/data.yaml` (full)
- Tests: `uv run pytest -q`
- N-gram baseline (the makemore bigram model generalized to order 4): `uv run python scripts/ngram_baseline.py` writes `results/ngram_baseline.json`. See `docs/ngram_baseline.md`.
- Scripts add `src/` to `sys.path` themselves, because macOS can hide uv's editable `.pth` file. For ad-hoc `python -c` commands, use `PYTHONPATH=src`.
- Deviations from the plan are logged in `results/deviations.md`.

Status, 2026-11-25: Gates 2-5 approved (docs/phase_2-5.md; deviations D4-D8). D8 done (colab_results/gate6). Finish the report in `report/main.tex` (Overleaf): figures, refs check, 2-page fit.

Colab from Gate 2 on: see `docs/colab_pipeline.md`. Drive (`MyDrive/unlearn/{checkpoints,results}`) is the working copy; the Mac receives gate packs in `colab_results/gateN/`. Rebuild the code bundle and notebooks with `uv run python scripts/make_colab.py`. Ready: notebooks 02 (Gate 2 eval), 03 (Stage 2), 04 (unlearning), 05 (test eval + relearning); each runs only after the previous gate is approved. Scope = deviation D5 (deadline 2026-11-25: 1 seed, 6 unlearning runs, 2 relearning conditions).
