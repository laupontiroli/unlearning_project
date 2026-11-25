"""Relearning attacks on θ_ref and the selected unlearned models (PLAN §9.3, D5).

    python scripts/relearn.py --seed 0
reads results/operating_points_s{seed}.json and runs, for θ_ref and the practitioner point of the
selected M1/M2/M3 runs, the conditions n0 and fresh1M. Finished runs are skipped.
Outputs results/relearn_{label}_{condition}_s{seed}.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

from unlearnlang.relearn import CONDITIONS, relearn
from unlearnlang.utils import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--with-oracle", action="store_true",
                    help="also relearn from the oracle point of the selected M3 run (starts at θ_ref's EN level; D7)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--points", default=None, help="operating-points file (default results/operating_points_s{seed}.json)")
    ap.add_argument("--suffix", default="", help="appended to the unlearned models' labels, e.g. proxy -> M3_proxy (D8)")
    ap.add_argument("--eval-every", type=int, default=25)
    a = ap.parse_args()
    pts = json.loads(Path(a.points or f"results/operating_points_s{a.seed}.json").read_text(encoding="utf-8"))
    models = {"theta_ref": f"checkpoints/ref_long_s{a.seed}.pt"}
    sfx = f"_{a.suffix}" if a.suffix else ""
    for m, run in pts["selected"].items():
        models[m + sfx] = pts["runs"][run]["practitioner"]["ckpt"]
    if a.with_oracle and pts["runs"][pts["selected"]["M3"]]["oracle"]:
        models[f"M3{sfx}_oracle"] = pts["runs"][pts["selected"]["M3"]]["oracle"]["ckpt"]
    device = pick_device()
    for cond in a.conditions:
        for label, ckpt in models.items():
            relearn(ckpt, label, cond, a.seed, device, steps=a.steps, eval_every=a.eval_every)


if __name__ == "__main__":
    main()
