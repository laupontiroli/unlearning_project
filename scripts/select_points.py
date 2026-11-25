"""Pick the practitioner and oracle operating points of every unlearning run (PLAN §8, D5).

    python scripts/select_points.py --seed 0 [--config configs/unlearn_v2.yaml]   ->  results/operating_points_s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

from unlearnlang.points import select_points


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default results/operating_points_s{seed}.json (D8 uses its own file)")
    ap.add_argument("--config", default=None, help="only the runs of this config's grid (e.g. configs/unlearn_v2.yaml)")
    a = ap.parse_args()
    names = None
    if a.config:
        import yaml
        from unlearnlang.unlearn import run_name
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        names = [run_name({**cfg, **r}, a.seed) for r in cfg["grid"]]
    out = select_points(Path(a.results), a.seed, names)
    p = Path(a.out) if a.out else Path(a.results) / f"operating_points_s{a.seed}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"θ_ref en_val = {out['ref_en_val_bpc']}")
    for run, r in out["runs"].items():
        pr, o = r["practitioner"], r["oracle"]
        print(f"{run:24s} div={r['diverged_at']}  practitioner step {pr['step']:4d} pt {pr['pt_val_bpc']:.3f} en {pr['en_val_bpc']:.3f}"
              f" | oracle " + (f"step {o['step']} pt {o['pt_val_bpc']:.3f} en {o['en_val_bpc']:.3f}" if o else "never reached"))
    print("selected:", out["selected"])
    print("wrote", p)


if __name__ == "__main__":
    main()
