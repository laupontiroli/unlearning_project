"""Run unlearning (PLAN §8, scope D5). Starts from θ_b; never touches the pre-Stage-2 model.

    python scripts/unlearn.py --grid --seed 0                       # all runs in configs/unlearn.yaml
    python scripts/unlearn.py --method M3 --lr 5e-5 --lam 1 --seed 0 # one run
Finished runs are skipped; an interrupted run restarts from scratch.
Outputs: results/unlearn_{run}.jsonl, checkpoints/unlearn/{run}/step_XXXX.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

import yaml

from unlearnlang.unlearn import unlearn
from unlearnlang.utils import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unlearn.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--method")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    grid = cfg.pop("grid")
    runs = grid if a.grid else [{"method": a.method, "lr": a.lr, "lam": a.lam}]
    if a.max_steps:
        cfg["max_steps"] = a.max_steps
    device = pick_device()
    for r in runs:
        print(unlearn({**cfg, **r}, a.seed, device), flush=True)


if __name__ == "__main__":
    main()
