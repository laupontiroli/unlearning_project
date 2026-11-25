"""Relearning attacks (PLAN §9.3, scope D5). Evaluation side: may load θ_ref.

Fine-tune a model for 300 steps (AdamW lr 1e-4 constant, weight_decay 0, grad clip 1, batch 32)
and track en_val / pt_val bpc every 25 steps on the same fixed batches as all other curves.
Conditions:
    n0        32 PT sequences per step (zero-example attack: does English come back on its own?)
    fresh1M   16 PT + 16 EN per step, EN drawn only from the first 1M characters of en_relearn_fresh
All models see the same data stream (same sampler seeds), so curves are paired.
No checkpoints are kept: only the curves matter.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path

import torch

from unlearnlang.data import BatchSampler, load_split
from unlearnlang.evaluate import load_model
from unlearnlang.train import FixedEval
from unlearnlang.utils import append_jsonl, autocast_dtype, seed_everything

LN2 = math.log(2)
CONDITIONS = {"n0": {"pt": 32, "en": 0, "en_chars": 0}, "fresh1M": {"pt": 16, "en": 16, "en_chars": 1_000_000}}


def relearn(ckpt: str, label: str, condition: str, seed: int, device: torch.device, data_dir: Path = Path("data"),
            results_dir: Path = Path("results"), steps: int = 300, lr: float = 1e-4, eval_every: int = 25,
            block_size: int = 512, eval_batches: int = 100, eval_batch_size: int = 16) -> Path:
    log = Path(results_dir) / f"relearn_{label}_{condition}_s{seed}.jsonl"
    if log.exists() and '"summary"' in log.read_text(encoding="utf-8"):
        print(f"{log.name}: already finished", flush=True)
        return log
    log.unlink(missing_ok=True)
    seed_everything(seed)
    c = CONDITIONS[condition]
    processed = data_dir / "processed"
    model = load_model(ckpt, device).train()
    samplers = [BatchSampler(load_split(processed, "pt_train"), block_size, c["pt"], seed * 1000 + 21, device)]
    if c["en"]:
        en = load_split(processed, "en_relearn_fresh")[: c["en_chars"]]
        samplers.append(BatchSampler(en, block_size, c["en"], seed * 1000 + 22, device))
    evaluator = FixedEval(processed, ["pt_val", "en_val"], eval_batches, eval_batch_size, block_size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    amp = autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if amp == torch.float16 else None
    autocast = (lambda: torch.autocast("cuda", dtype=amp)) if amp else nullcontext

    base = {"label": label, "condition": condition, "ckpt": ckpt}
    append_jsonl(log, {**base, "step": 0, **evaluator.all(model)})
    for step in range(1, steps + 1):
        xs, ys = zip(*(s.next() for s in samplers))
        with autocast():
            _, loss = model(torch.cat(xs), torch.cat(ys))
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        if step % eval_every == 0:
            ev = evaluator.all(model)
            append_jsonl(log, {**base, "step": step, "train_bpc": round(loss.item() / LN2, 5), **ev})
            print(f"{label} {condition} step {step} | {ev}", flush=True)
    append_jsonl(log, {"summary": {**base, "steps": steps}})
    return log
