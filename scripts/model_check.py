"""Gate 1 report: parameter count, initial loss vs ln(V), overfit curve on one real batch, untrained sample.

Usage: uv run python scripts/model_check.py [--data-dir data] [--device auto|cpu|mps|cuda]
Output: results/gate1.json, results/figures/overfit_curve.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from unlearnlang.data import BatchSampler, Vocab, load_split
from unlearnlang.model import GPT, ModelConfig
from unlearnlang.utils import pick_device, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    seed_everything(0)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    d = Path(args.data_dir)
    vocab = Vocab.load(d / "vocab.json")
    cfg = ModelConfig(vocab_size=len(vocab))
    model = GPT(cfg).to(device)
    V = cfg.vocab_size

    n = model.num_params()
    formula = 6 * 12 * cfg.d_model ** 2 + cfg.block_size * cfg.d_model + V * cfg.d_model
    sampler = BatchSampler(load_split(d / "processed", "pt_train"), cfg.block_size, 8, seed=0, device=device)
    x, y = sampler.next()

    model.eval()
    with torch.no_grad():
        _, loss0 = model(x, y)

    # overfit ONE small real batch (dropout off) -> loss should approach 0
    model_of = GPT(ModelConfig(vocab_size=V, dropout=0.0)).to(device)
    xb, yb = x[:4, :128], y[:4, :128]
    opt = torch.optim.AdamW(model_of.parameters(), lr=1e-3)
    curve, t0 = [], time.time()
    for step in range(args.steps):
        _, loss = model_of(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(loss.item())
    secs = time.time() - t0

    prompt = torch.from_numpy(vocab.encode("o brasil é ")).long().unsqueeze(0)  # (1, T0)
    sample = model.cpu().generate(prompt, 120, temperature=1.0, generator=torch.Generator().manual_seed(0))

    report = {
        "device": str(device), "params": n, "formula": formula, "param_rel_diff": (n - formula) / formula,
        "initial_loss_nats": loss0.item(), "ln_V": math.log(V),
        "initial_bpc": loss0.item() / math.log(2), "log2_V": math.log2(V),
        "overfit_loss_first_last": [curve[0], curve[-1]], "overfit_steps": args.steps, "overfit_seconds": round(secs, 1),
        "untrained_sample": vocab.decode(sample[0].tolist()),
    }
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/gate1.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    plt.figure(figsize=(5, 3.2))
    plt.plot(curve)
    plt.axhline(math.log(V), ls="--", c="k", lw=1, label="ln V (uniform)")
    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("train loss (nats)")
    plt.title("Overfitting one batch (4 x 128 chars)"); plt.legend(); plt.tight_layout()
    plt.savefig("results/figures/overfit_curve.png", dpi=130)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
