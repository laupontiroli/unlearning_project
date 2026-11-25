"""Unlearning English from θ_b (PLAN §8, scope D5).

Allowed inputs: θ_b weights, en_forget_train (or, for D8, en_relearn_fresh from character 1M on: proxy English
no model has trained on), pt_train, pt_val, en_val. The pre-Stage-2 model is
FORBIDDEN here in any form: this module and everything it imports must not name it
(tests/test_isolation.py checks the source), and init_from is refused if its file name says so.

Methods (L_x = mean next-character cross-entropy on batch x, in nats):
    M1  retain-only control   L_PT
    M2  gradient ascent       -L_EN
    M3  NegGrad+              -L_EN + lam * L_PT
Each step uses 32 EN + 32 PT sequences; AdamW(weight_decay=0), constant lr, grad clip 1.0,
fresh optimizer. Every eval_interval steps (and every dense_interval steps up to dense_until): pt_val/en_val bpc on the same fixed batches as Stages
1 and 2, plus a weights-only checkpoint. Divergence guard: stop when L_EN on the training batch
exceeds ln(V) + 1 nats (worse than uniform guessing) and save the last good weights.
"""
from __future__ import annotations

import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from unlearnlang.data import BatchSampler, Vocab, load_split
from unlearnlang.model import GPT, ModelConfig
from unlearnlang.train import FixedEval
from unlearnlang.utils import append_jsonl, autocast_dtype, load_checkpoint, save_checkpoint, seed_everything

LN2 = math.log(2)
METHODS = ("M1", "M2", "M3")
FORGET_SPLITS = ("en_forget_train", "en_relearn_fresh")
RESERVED_FRESH_CHARS = 1_000_000   # en_relearn_fresh[:1M] belongs to the relearning attack (D8)


def forget_data(processed: Path, cfg: dict):
    """English used for unlearning: en_forget_train (seen pool) or proxy text the model never saw (D8)."""
    split, offset = cfg.get("forget_split", "en_forget_train"), int(cfg.get("forget_offset", 0))
    if split not in FORGET_SPLITS:
        raise ValueError(f"forget_split must be one of {FORGET_SPLITS}, got {split}")
    if split == "en_relearn_fresh" and offset < RESERVED_FRESH_CHARS:
        raise ValueError(f"en_relearn_fresh[:{RESERVED_FRESH_CHARS}] is reserved for relearning; use forget_offset >= that")
    return load_split(processed, split)[offset:]


def unlearn_loss(method: str, l_en: torch.Tensor, l_pt: torch.Tensor, lam: float) -> torch.Tensor:
    if method == "M1":
        return l_pt
    if method == "M2":
        return -l_en
    if method == "M3":
        return -l_en + lam * l_pt
    raise ValueError(f"unknown method {method}")


def run_name(cfg: dict, seed: int) -> str:
    name = f"{cfg['method']}_lr{cfg['lr']:g}"
    if cfg["method"] == "M3":
        name += f"_lam{cfg['lam']:g}"
    if cfg.get("tag"):
        name += f"_{cfg['tag']}"
    return f"{name}_s{seed}"


def is_eval_step(step: int, cfg: dict) -> bool:
    """Every eval_interval steps, plus every dense_interval steps up to dense_until (D6: GA-type runs
    can collapse within ~50 steps, so the early trajectory needs finer resolution)."""
    dense = cfg.get("dense_until", 0) and step <= cfg["dense_until"] and step % cfg["dense_interval"] == 0
    return bool(step % cfg["eval_interval"] == 0 or step == cfg["max_steps"] or dense)


def check_init(path: str) -> None:
    """The starting point must be a Stage-2 model; refuse anything named like the reference."""
    name = Path(path).name.lower()
    if name.startswith("ref") or "stage1" in str(path).lower():
        raise ValueError(f"unlearning may not start from {path} (§11 rule 4)")


def is_done(log_path: Path) -> bool:
    return log_path.exists() and '"summary"' in log_path.read_text(encoding="utf-8")


def unlearn(cfg: dict, seed: int, device: torch.device, ckpt_dir: Path = Path("checkpoints"),
            results_dir: Path = Path("results")) -> dict:
    assert cfg["method"] in METHODS, cfg["method"]
    seed_everything(seed)
    run = run_name(cfg, seed)
    log_path = Path(results_dir) / f"unlearn_{run}.jsonl"
    run_dir = Path(ckpt_dir) / "unlearn" / run
    if is_done(log_path):
        print(f"{run}: already finished", flush=True)
        return {"run": run, "skipped": True}
    if log_path.exists() or run_dir.exists():   # an interrupted run restarts from scratch (same seed)
        print(f"{run}: removing an unfinished attempt and restarting", flush=True)
        log_path.unlink(missing_ok=True)
        shutil.rmtree(run_dir, ignore_errors=True)

    init = cfg["init_from"].format(seed=seed)
    check_init(init)
    ck = load_checkpoint(init)
    model_cfg = ck["config"]["model"]
    model = GPT(ModelConfig(vocab_size=ck["vocab_size"], **model_cfg))
    model.load_state_dict(ck["model"])
    model.to(device).train()

    processed = Path(cfg["data_dir"]) / "processed"
    V = len(Vocab.load(Path(cfg["data_dir"]) / "vocab.json"))
    guard = math.log(V) + cfg.get("guard_margin_nats", 1.0)
    en = BatchSampler(forget_data(processed, cfg), cfg["block_size"], cfg["en_batch"], seed * 1000 + 7, device)
    pt = BatchSampler(load_split(processed, "pt_train"), cfg["block_size"], cfg["pt_batch"], seed * 1000 + 8, device)
    evaluator = FixedEval(processed, ["pt_val", "en_val"], cfg["eval_batches"], cfg["eval_batch_size"],
                          cfg["block_size"], device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=0.0)
    amp_dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else nullcontext
    saved_cfg = {**cfg, "model": model_cfg, "init_from": init, "run": run}

    def save(step: int, state: dict | None = None) -> str:
        p = run_dir / f"step_{step:04d}.pt"
        save_checkpoint(p, {"model": state or model.state_dict(), "config": saved_cfg, "seed": seed,
                            "step": step, "vocab_size": V})
        return str(p)

    ev = evaluator.all(model)
    append_jsonl(log_path, {"run": run, "step": 0, "ckpt": init, **ev})
    print(f"{run} step 0 | {ev}", flush=True)
    t0, acc = time.time(), {"l_en": 0.0, "l_pt": 0.0, "n": 0}
    prev_state, diverged, step = None, None, 0
    while step < cfg["max_steps"]:
        xe, ye = en.next()
        xp, yp = pt.next()
        with autocast():
            _, l_en = model(xe, ye)
            _, l_pt = model(xp, yp)
        if l_en.item() > guard:
            diverged = {"event": "diverged", "step": step, "l_en_nats": round(l_en.item(), 4), "guard_nats": round(guard, 4)}
            break
        loss = unlearn_loss(cfg["method"], l_en, l_pt, cfg.get("lam", 1.0))
        prev_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
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
        acc["l_en"] += l_en.item(); acc["l_pt"] += l_pt.item(); acc["n"] += 1
        if is_eval_step(step, cfg):
            ev = evaluator.all(model)
            rec = {"run": run, "step": step, "ckpt": save(step), "train_en_bpc": round(acc["l_en"] / acc["n"] / LN2, 5),
                   "train_pt_bpc": round(acc["l_pt"] / acc["n"] / LN2, 5), "grad_norm": round(float(gnorm), 4),
                   "sec_per_step": round((time.time() - t0) / acc["n"], 4), **ev}
            append_jsonl(log_path, rec)
            print(f"{run} step {step} | " + " ".join(f"{k}={v}" for k, v in rec.items() if k not in ("run", "ckpt")), flush=True)
            t0, acc = time.time(), {"l_en": 0.0, "l_pt": 0.0, "n": 0}

    if diverged:
        append_jsonl(log_path, {"run": run, **diverged})
        print(f"{run} DIVERGED at step {step}: {diverged}", flush=True)
        last_good = step - 1
        if last_good >= 1 and not is_eval_step(last_good, cfg):   # not already saved + evaluated
            model.load_state_dict(prev_state)
            ev = evaluator.all(model)
            append_jsonl(log_path, {"run": run, "step": last_good, "ckpt": save(last_good, prev_state),
                                    "last_good": True, **ev})
    summary = {"run": run, "method": cfg["method"], "lr": cfg["lr"], "lam": cfg.get("lam") if cfg["method"] == "M3" else None,
               "steps": step, "diverged_at": diverged["step"] if diverged else None}
    append_jsonl(log_path, {"summary": summary})
    return summary
