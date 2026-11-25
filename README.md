# Unlearning a Concept Without Its Training Data

Removing English from a character-level Portuguese transformer: Laura Pontiroli Machado, Computer Science, Insper.

**Report (Overleaf, read-only):** https://www.overleaf.com/project/6b074bc64156c5154a683973/share#16834e4425e10ad218ec9f88e7a30d68bd1c879907f9041d
The LaTeX source is in [`report/`](report/).

## Idea

A small transformer is trained on Portuguese Wikipedia only (θ_ref). It is then trained further on a 50/50 Portuguese–English mix (θ_b). Unlearning English from θ_b should lead back to θ_ref. That makes θ_ref an exact reference that needs no retraining, and the unlearning code never loads it.

I compare three methods:
- M1: a Portuguese-only control.
- M2: gradient ascent.
- M3: NegGrad+.

M3 is also run with *proxy* English that no model has seen. All results use bits per character, KL to θ_ref, English leakage in generated text, and relearning attacks.

**Main findings (seed 0):**
- No method reaches θ_ref.
- Gradient ascent collapses within about 20 steps.
- NegGrad+ closes 54% of the English gap, but by KL it is further from θ_ref than the control.
- Unseen English unlearns exactly as well as training-pool English.
- In every case, 25 steps of fine-tuning on Portuguese alone bring most of the English back: it was suppressed, not removed.

## Layout

| Path | Contents |
|---|---|
| `src/unlearnlang/` | data, model (nanoGPT-style), training, unlearning, evaluation, relearning |
| `scripts/` | command-line entry points (data prep, training, unlearning, evaluation, plots, Colab builder) |
| `configs/` | YAML configs for every run |
| `tests/` | pytest suite, including the isolation test that keeps θ_ref out of the unlearning code |
| `colab/` | Colab notebooks 02–06 (one per phase) |
| `results/`, `colab_results/` | logs, tables, figures and generated samples (gate packs 5 and 6) |
| `docs/` | research plan (`PLAN.md`), phase notes, Colab pipeline |
| `results/deviations.md` | every deviation from the plan (D1–D8) |

## Reproduce

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv). Data and checkpoints are not in the repository.

```bash
uv sync
uv run pytest -q
HF_HOME=$PWD/data/raw/hf uv run python scripts/prepare_data.py --config configs/data.yaml
uv run python scripts/make_colab.py   # builds colab/code_bundle.tar.gz and the notebooks
```

Training, unlearning and evaluation ran on a Colab T4. The steps are described in [`docs/colab_pipeline.md`](docs/colab_pipeline.md). Run the notebooks in `colab/` in order (02 → 06).
