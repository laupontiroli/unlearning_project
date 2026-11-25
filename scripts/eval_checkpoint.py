"""Evaluate one checkpoint: bpc on 500 fixed chunks (§9.1) + 200 generations per language (§9.2).

Usage:
    uv run python scripts/eval_checkpoint.py --ckpt checkpoints/ref_s0.pt --name ref_s0 \
        --splits pt_val en_val --test-splits pt_test --prompts val

--test-splits may be used only ONCE per (checkpoint, split): every access is logged in
results/test_access.jsonl and a second access is refused (PLAN §11 rule 5).
--prompts val|test chooses where the generation prompts come from (test only in Phase 5).
Outputs: results/eval.jsonl (one line per call), results/samples/{name}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

import numpy as np
import torch

from unlearnlang.data import load_split
from unlearnlang.evaluate import bootstrap_ci, bpc_per_chunk, chunks, generation_report, kl_per_chunk, load_model
from unlearnlang.utils import append_jsonl, pick_device, sha256_file

TEST_LOG = Path("results/test_access.jsonl")


def check_test_access(ckpt_sha: str, split: str) -> None:
    if TEST_LOG.exists():
        for line in TEST_LOG.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["ckpt_sha256"] == ckpt_sha and r["split"] == split:
                raise RuntimeError(f"{split} was already evaluated for this checkpoint at {r['time']}; refusing (§11.5)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", nargs="*", default=["pt_val", "en_val"])
    ap.add_argument("--test-splits", nargs="*", default=[])
    ap.add_argument("--prompts", choices=["val", "test", "none"], default="val")
    ap.add_argument("--n-chunks", type=int, default=500)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--kl-ref", default=None, help="reference checkpoint for KL(p_ref || p_model) on every split")
    args = ap.parse_args()
    device = pick_device()
    d = Path(args.data_dir)
    sha = sha256_file(args.ckpt)
    for s in args.test_splits:
        check_test_access(sha, s)
    model = load_model(args.ckpt, device)
    T = model.cfg.block_size
    t0 = time.time()

    ref = load_model(args.kl_ref, device) if args.kl_ref else None
    rec = {"name": args.name, "ckpt": args.ckpt, "ckpt_sha256": sha, "n_chunks": args.n_chunks, "bpc": {}, "bpc_ci95": {}}
    if ref is not None:
        rec.update(kl_ref=args.kl_ref, kl_ref_sha256=sha256_file(args.kl_ref), kl_bits={}, kl_ci95={})
    per_chunk = {}
    for s in args.splits + args.test_splits:
        ch = chunks(np.asarray(load_split(d / "processed", s)), args.n_chunks, T)
        v = bpc_per_chunk(model, ch, device)
        per_chunk[f"bpc_{s}"] = v
        rec["bpc"][s] = round(float(v.mean()), 5)
        rec["bpc_ci95"][s] = [round(x, 5) for x in bootstrap_ci(v)]
        if ref is not None:
            k = kl_per_chunk(ref, model, ch, device)
            per_chunk[f"kl_{s}"] = k
            rec["kl_bits"][s] = round(float(k.mean()), 5)
            rec["kl_ci95"][s] = [round(x, 5) for x in bootstrap_ci(k)]
        if s in args.test_splits:
            append_jsonl(TEST_LOG, {"ckpt": args.ckpt, "ckpt_sha256": sha, "split": s, "bpc": rec["bpc"][s]})
    # per-chunk values, for paired comparisons between models in the report (same chunks for every model)
    Path("results/chunks").mkdir(parents=True, exist_ok=True)
    np.savez(f"results/chunks/{args.name}.npz", **per_chunk)
    print(f"bpc: {rec['bpc']}", flush=True)

    if args.prompts != "none":
        wl = json.loads((d / "wordlists.json").read_text(encoding="utf-8"))
        splits = {"pt": f"pt_{args.prompts}", "en": f"en_{args.prompts}"}
        metrics, samples = generation_report(model, d, wl, device, splits, n=args.n_prompts)
        rec["generation"] = {"prompt_splits": splits, **metrics}
        out = Path("results/samples") / f"{args.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"generation: {json.dumps(metrics)}", flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    append_jsonl("results/eval.jsonl", rec)


if __name__ == "__main__":
    main()
