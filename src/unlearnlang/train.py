"""Generic training loop for Stage 1 (PT only) and Stage 2 (mixed PT + EN). PLAN §6, §7.

Features: AdamW with weight decay on matrices only, linear warmup + cosine decay, grad clipping,
fixed-batch evaluation (same chunks every time), atomic checkpoints (keep last 2 + best by pt_val)
with full RNG state, --resume, JSONL logging, and a read-only protected final checkpoint (SHA-256 record).
"""
from __future__ import annotations

import json
import math
import os
import random
import stat
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from unlearnlang.data import BatchSampler, Vocab, fixed_offsets, load_split
from unlearnlang.model import GPT, ModelConfig
from unlearnlang.utils import (append_jsonl, autocast_dtype, load_checkpoint, save_checkpoint,
                               seed_everything, sha256_file)

LN2 = math.log(2)


# ----------------------------------------------------------------------------- config
def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["model"] = yaml.safe_load(Path(cfg["model_config"]).read_text(encoding="utf-8"))
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    cfg["model"]["block_size"] = max(cfg["model"]["block_size"], cfg["block_size"])
    return cfg


def build_model(cfg: dict, vocab_size: int) -> GPT:
    return GPT(ModelConfig(vocab_size=vocab_size, **cfg["model"]))


def lr_at(step: int, cfg: dict) -> float:
    """Linear warmup to lr, then cosine decay to min_lr at max_steps."""
    lr, min_lr, warm, total = cfg["lr"], cfg["min_lr"], cfg["warmup_steps"], cfg["max_steps"]
    if step < warm:
        return lr * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, total - warm))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, cfg: dict) -> torch.optim.AdamW:
    """Weight decay on >=2-D tensors (matrices, embeddings); none on biases and LayerNorm."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=tuple(cfg["betas"]))


# ------------------------------------------------------------------------- evaluation
class FixedEval:
    """Same chunks of each split every time, so eval curves are comparable across steps and runs."""

    def __init__(self, processed: Path, splits: list[str], n_batches: int, batch_size: int,
                 block_size: int, device: torch.device, seed: int = 1234):
        self.device, self.block_size, self.batch_size = device, block_size, batch_size
        self.data = {s: load_split(processed, s) for s in splits}
        self.offsets = {s: fixed_offsets(len(d), block_size, n_batches * batch_size, seed)
                        for s, d in self.data.items()}

    @torch.no_grad()
    def bpc(self, model: GPT, split: str) -> float:
        was_training = model.training
        model.eval()
        d, T, B = self.data[split], self.block_size, self.batch_size
        offs = self.offsets[split]
        total, n = 0.0, 0
        for i in range(0, len(offs), B):
            chunk = np.stack([np.asarray(d[o:o + T + 1]) for o in offs[i:i + B]]).astype(np.int64)
            t = torch.from_numpy(chunk).to(self.device)
            _, loss = model(t[:, :-1], t[:, 1:])
            total += loss.item() * t.size(0)
            n += t.size(0)
        model.train(was_training)
        return total / n / LN2

    def all(self, model: GPT) -> dict[str, float]:
        return {f"{s}_bpc": round(self.bpc(model, s), 5) for s in self.data}


# --------------------------------------------------------------------------- training
class MixedSampler:
    """Each batch concatenates `n` sequences from each split in cfg['mixture'] (e.g. 32 PT + 32 EN)."""

    def __init__(self, processed: Path, mixture: dict[str, int], block_size: int, seed: int, device):
        self.samplers = {s: BatchSampler(load_split(processed, s), block_size, n, seed=seed * 1000 + i, device=device)
                         for i, (s, n) in enumerate(sorted(mixture.items()))}

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = zip(*(s.next() for s in self.samplers.values()))
        return torch.cat(xs), torch.cat(ys)

    def state_dict(self) -> dict:
        return {k: s.state_dict() for k, s in self.samplers.items()}

    def load_state_dict(self, sd: dict) -> None:
        for k, s in self.samplers.items():
            s.load_state_dict(sd[k])


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(sd: dict) -> None:
    random.setstate(sd["python"])
    np.random.set_state(sd["numpy"])
    torch.set_rng_state(sd["torch"])


def protect(path: Path, hashes_file: Path, key: str) -> str:
    """Make the file read-only and record its SHA-256. Refuses to change an existing record.

    Some filesystems (e.g. Google Drive mounted in Colab) do not support chmod; then the SHA-256
    record is the protection, and the file is made read-only after it is copied back locally.
    """
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError as e:
        print(f"WARNING: could not make {path} read-only ({e}); relying on the SHA-256 record", flush=True)
    digest = sha256_file(path)
    hashes = json.loads(hashes_file.read_text(encoding="utf-8")) if hashes_file.exists() else {}
    if key in hashes and hashes[key] != digest:
        raise RuntimeError(f"{key} already recorded with a different SHA; refusing to overwrite")
    hashes[key] = digest
    hashes_file.parent.mkdir(parents=True, exist_ok=True)
    hashes_file.write_text(json.dumps(hashes, indent=1), encoding="utf-8")
    return digest


def train(cfg: dict, seed: int, device: torch.device, resume: bool = False,
          ckpt_dir: Path = Path("checkpoints"), timing_only: bool = False, stop_at: int | None = None,
          results_dir: Path = Path("results")) -> dict:
    """Run one training job. Returns a summary dict.

    stop_at: save last.pt and return at this step without finishing (simulates an interruption;
             continue later with resume=True).
    """
    assert cfg["max_steps"], "max_steps must be set (run the timing run first)"
    seed_everything(seed)
    run = f"{cfg['name']}_s{seed}"
    run_dir = ckpt_dir / run
    final_path = ckpt_dir / f"{cfg['final_name']}_s{seed}.pt"
    log_path = Path(results_dir) / f"train_{run}.jsonl"
    if final_path.exists() and not timing_only:
        raise FileExistsError(f"{final_path} exists; refusing to overwrite")

    processed = Path(cfg["data_dir"]) / "processed"
    vocab = Vocab.load(Path(cfg["data_dir"]) / "vocab.json")
    model = build_model(cfg, len(vocab))
    if cfg.get("init_from"):
        model.load_state_dict(load_checkpoint(cfg["init_from"].format(seed=seed))["model"])
    model.to(device)
    opt = make_optimizer(model, cfg)
    sampler = MixedSampler(processed, cfg["mixture"], cfg["block_size"], seed, device)
    evaluator = FixedEval(processed, cfg["eval_splits"], cfg["eval_batches"], cfg["eval_batch_size"],
                          cfg["block_size"], device)
    amp_dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else nullcontext

    step, best = 0, float("inf")
    last = run_dir / "last.pt"
    if resume and last.exists():
        ck = load_checkpoint(last)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sampler.load_state_dict(ck["sampler"])
        set_rng_state(ck["rng"])
        step, best = ck["step"], ck["best_pt_val"]
        print(f"resumed {run} at step {step}", flush=True)

    def save(path: Path) -> None:
        save_checkpoint(path, {"model": model.state_dict(), "optimizer": opt.state_dict(), "sampler": sampler.state_dict(),
                               "rng": rng_state(), "step": step, "best_pt_val": best, "config": cfg, "seed": seed,
                               "vocab_size": len(vocab)})

    model.train()
    t_start, t_log, loss_acc, n_acc = time.time(), time.time(), 0.0, 0
    if not resume or step == 0:
        if not timing_only:
            ev = evaluator.all(model)
            append_jsonl(log_path, {"run": run, "step": 0, **ev})
            print(f"step 0 | {ev}", flush=True)
    while step < cfg["max_steps"]:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = sampler.next()
        with autocast():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        step += 1
        loss_acc += loss.item()
        n_acc += 1

        if timing_only:
            if step == 10:  # skip the first steps (MPS/CUDA kernel warm-up)
                t_after_warmup = time.time()
            continue
        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"]:
            dt = time.time() - t_log
            ev = evaluator.all(model)
            rec = {"run": run, "step": step, "lr": lr, "train_loss_bpc": round(loss_acc / n_acc / LN2, 5),
                   "grad_norm": round(float(gnorm), 4), "sec_per_step": round(dt / n_acc, 4), **ev}
            append_jsonl(log_path, rec)
            print(f"step {step} | " + " ".join(f"{k}={v}" for k, v in rec.items() if k != "run"), flush=True)
            loss_acc, n_acc, t_log = 0.0, 0, time.time()
            if ev.get("pt_val_bpc", float("inf")) < best:
                best = ev["pt_val_bpc"]
                save(run_dir / "best.pt")
        if step % cfg["ckpt_interval"] == 0 or step == cfg["max_steps"] or step == stop_at:
            if last.exists():
                os.replace(last, run_dir / "prev.pt")   # keep the last 2
            save(last)
        if step == stop_at:
            return {"run": run, "stopped_at": step}

    elapsed = time.time() - t_start
    summary = {"run": run, "steps": step, "seconds": round(elapsed, 1), "steps_per_sec": round(step / elapsed, 3),
               "device": str(device)}
    if timing_only:
        timed = step - 10
        summary["steps_per_sec_after_warmup"] = round(timed / (time.time() - t_after_warmup), 3)
        t0 = time.time()
        evaluator.all(model)
        summary["seconds_per_eval"] = round(time.time() - t0, 1)
        return summary

    final = {"model": model.state_dict(), "config": cfg, "seed": seed, "step": step, "vocab_size": len(vocab)}
    save_checkpoint(final_path, final)
    if cfg.get("protect_final"):
        summary["sha256"] = protect(final_path, Path(results_dir) / cfg.get("hashes_file", f"{cfg['final_name']}_hashes.json"),
                                    final_path.name)
    summary["final_checkpoint"] = str(final_path)
    append_jsonl(log_path, {"summary": summary})
    return summary
