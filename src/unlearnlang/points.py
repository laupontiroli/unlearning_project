"""Operating points of the unlearning runs (PLAN §8). Evaluation side: may read θ_ref's numbers.

- practitioner: the last checkpoint before pt_val bpc worsens by more than 2% relative to θ_b
  (the run's step 0), or the final step. Uses no reference information.
- oracle (best case, labelled as such): the first checkpoint whose en_val bpc reaches θ_ref's,
  read from the final eval of the Stage-1 log (same fixed eval batches). None if never reached.
- selected run per method (D5): the only run for M1/M2; for M3 the run with the highest en_val
  at its practitioner point (validation data only).
"""
from __future__ import annotations

import json
from pathlib import Path

PT_TOLERANCE = 0.02


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def reference_en_val(results: Path, seed: int, ref_run: str = "stage1_long") -> float:
    recs = [r for r in read_jsonl(results / f"train_{ref_run}_s{seed}.jsonl") if "en_val_bpc" in r]
    return recs[-1]["en_val_bpc"]


def trajectory(log: list[dict]) -> list[dict]:
    """Eval records sorted by step, without any model at or after a divergence."""
    div = next((r["step"] for r in log if r.get("event") == "diverged"), None)
    recs = [r for r in log if "pt_val_bpc" in r and (div is None or r["step"] < div)]
    return sorted(recs, key=lambda r: r["step"])


def operating_points(traj: list[dict], ref_en: float) -> dict:
    base = traj[0]["pt_val_bpc"]
    prac = traj[0]
    for r in traj:
        if r["pt_val_bpc"] > (1 + PT_TOLERANCE) * base:
            break
        prac = r
    oracle = next((r for r in traj[1:] if r["en_val_bpc"] >= ref_en), None)
    keep = ("step", "ckpt", "pt_val_bpc", "en_val_bpc")
    return {"practitioner": {k: prac[k] for k in keep},
            "oracle": {k: oracle[k] for k in keep} if oracle else None,
            "theta_b": {k: traj[0][k] for k in keep}}


def select_points(results: Path, seed: int, run_names: list[str] | None = None) -> dict:
    """run_names: only these runs (e.g. the grid of one config); default: every unlearn_*_s{seed}.jsonl."""
    ref_en = reference_en_val(results, seed)
    runs = {}
    paths = ([results / f"unlearn_{n}.jsonl" for n in run_names] if run_names
             else sorted(results.glob(f"unlearn_*_s{seed}.jsonl")))
    for p in paths:
        log = read_jsonl(p)
        summ = next((r["summary"] for r in log if "summary" in r), None)
        if summ is None:
            raise RuntimeError(f"{p.name} is unfinished; finish or rerun it first")
        runs[summ["run"]] = {**summ, **operating_points(trajectory(log), ref_en)}
    selected = {}
    for m in ("M1", "M2", "M3"):
        cands = [r for r in runs.values() if r["method"] == m]
        if cands:
            best = max(cands, key=lambda r: (r["practitioner"]["en_val_bpc"], -r["practitioner"]["pt_val_bpc"]))
            selected[m] = best["run"]
    return {"seed": seed, "ref_en_val_bpc": ref_en, "pt_tolerance": PT_TOLERANCE, "runs": runs, "selected": selected}
