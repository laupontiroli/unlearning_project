"""Phase 5 evaluation on the test splits (PLAN §9.1–9.2, D5): every model once.

    python scripts/phase5_eval.py --seed 0 [--dry-run]

Models: θ_ref (en_test only: its pt_test was read once at Gate 2), θ_b, θ_en, and every unlearning
run at its practitioner point, plus the oracle point of the selected run per method. Checkpoints
that are the same file are evaluated once. KL is to θ_ref. Prompts come from the test splits.
Names already present in results/eval.jsonl are skipped, so the script can be re-run after a disconnect.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def plan(seed: int, points: str | None = None) -> list[tuple[str, str, list[str], bool]]:
    """(name, ckpt, test_splits, with_kl)"""
    ref = f"checkpoints/ref_long_s{seed}.pt"
    jobs = [(f"theta_ref_s{seed}@test", ref, ["en_test"], False),
            (f"theta_b_s{seed}@test", f"checkpoints/theta_b_s{seed}.pt", ["pt_test", "en_test"], True),
            (f"theta_en_s{seed}@test", f"checkpoints/theta_en_s{seed}.pt", ["pt_test", "en_test"], True)]
    pts = json.loads(Path(points or f"results/operating_points_s{seed}.json").read_text(encoding="utf-8"))
    seen = {j[1] for j in jobs}
    for run, info in pts["runs"].items():
        points = [("prac", info["practitioner"])]
        if run in pts["selected"].values() and info["oracle"]:
            points.append(("oracle", info["oracle"]))
        for tag, p in points:
            if p["ckpt"] in seen:
                print(f"{run}@{tag}: same checkpoint as an earlier job ({p['ckpt']}), not evaluated again")
                continue
            seen.add(p["ckpt"])
            jobs.append((f"{run}@{tag}", p["ckpt"], ["pt_test", "en_test"], True))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--points", default=None, help="operating-points file (D8: results/operating_points_proxy_s0.json)")
    a = ap.parse_args()
    ref = f"checkpoints/ref_long_s{a.seed}.pt"
    done = set()
    if Path("results/eval.jsonl").exists():
        done = {json.loads(l)["name"] for l in Path("results/eval.jsonl").read_text(encoding="utf-8").splitlines()}
    for name, ckpt, tests, kl in plan(a.seed, a.points):
        if name in done:
            print(f"{name}: already evaluated"); continue
        cmd = [sys.executable, "scripts/eval_checkpoint.py", "--ckpt", ckpt, "--name", name, "--splits",
               "--test-splits", *tests, "--prompts", "test"] + (["--kl-ref", ref] if kl else [])
        print(" ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
