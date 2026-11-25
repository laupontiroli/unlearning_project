"""Train Stage 1 / Stage 2 models.

Usage:
    uv run python scripts/train.py --config configs/stage1.yaml --seed 0 [--max-steps N] [--resume]
    uv run python scripts/train.py --config configs/stage1.yaml --seed 0 --timing      # 110 steps, no eval/ckpt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

import torch

from unlearnlang.train import load_config, train
from unlearnlang.utils import append_jsonl, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-dir", default="checkpoints", help="e.g. a Google Drive folder on Colab")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    cfg = load_config(args.config, {"max_steps": 110 if args.timing else args.max_steps})
    summary = train(cfg, args.seed, device, resume=args.resume, timing_only=args.timing,
                    ckpt_dir=Path(args.ckpt_dir), results_dir=Path(args.results_dir))
    if args.timing:
        tokens_per_step = sum(cfg["mixture"].values()) * cfg["block_size"]
        summary.update({"config": args.config, "chars_per_step": tokens_per_step,
                        "eval_interval": cfg["eval_interval"]})
        append_jsonl(Path(args.results_dir) / "timing.jsonl", summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
