"""Colab helpers (docs/colab_pipeline.md).

    python scripts/colab_sync.py merge --src results_mac --dst results   # Mac results -> Drive, never overwrites
    python scripts/colab_sync.py verify                                     # re-hash protected checkpoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

from unlearnlang.colab import merge_results, verify_hashes


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--src", default="results_mac")
    m.add_argument("--dst", default="results")
    v = sub.add_parser("verify")
    v.add_argument("--results", default="results")
    v.add_argument("--ckpt-dir", default="checkpoints")
    a = ap.parse_args()
    lines = merge_results(Path(a.src), Path(a.dst)) if a.cmd == "merge" else verify_hashes(Path(a.results), Path(a.ckpt_dir))
    print("\n".join(lines) or "nothing to do")


if __name__ == "__main__":
    main()
