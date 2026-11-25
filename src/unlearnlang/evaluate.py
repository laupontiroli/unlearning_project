"""Evaluation: bpc on fixed chunks, KL to a reference, generation + language leakage (PLAN §9).

This module MAY load θ_ref (it is the evaluator). unlearn.py must never import it (§11 rule 4).
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from unlearnlang.data import Vocab, fixed_offsets, load_split
from unlearnlang.metrics import leakage, make_prompts
from unlearnlang.model import GPT, ModelConfig
from unlearnlang.utils import load_checkpoint

LN2 = math.log(2)
EVAL_SEED = 4321          # chunk offsets for §9.1 (different from the training-time FixedEval seed)
PROMPT_SEEDS = {"pt": 11, "en": 12}


def load_model(path: str | Path, device: torch.device) -> GPT:
    ck = load_checkpoint(path)
    mcfg = dict(ck["config"]["model"])
    model = GPT(ModelConfig(vocab_size=ck["vocab_size"], **mcfg))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def chunks(data: np.ndarray, n: int, block_size: int, seed: int = EVAL_SEED) -> np.ndarray:
    """(n, block_size+1) fixed chunks of a split."""
    offs = fixed_offsets(len(data), block_size, n, seed)
    return np.stack([np.asarray(data[o:o + block_size + 1]) for o in offs]).astype(np.int64)


@torch.no_grad()
def bpc_per_chunk(model: GPT, ch: np.ndarray, device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Mean bits per character of each chunk (all chunks have the same length)."""
    out = []
    for i in range(0, len(ch), batch_size):
        t = torch.from_numpy(ch[i:i + batch_size]).to(device)
        logits = model(t[:, :-1])[0].float()
        nll = F.cross_entropy(logits.transpose(1, 2), t[:, 1:], reduction="none")   # (B, T)
        out.append((nll.mean(1) / LN2).cpu().numpy())
    return np.concatenate(out)


def bpc(model: GPT, ch: np.ndarray, device: torch.device, batch_size: int = 16) -> float:
    return float(bpc_per_chunk(model, ch, device, batch_size).mean())


@torch.no_grad()
def kl_per_chunk(ref: GPT, model: GPT, ch: np.ndarray, device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Per-chunk mean per-character KL(p_ref || p_model) in bits, teacher-forced."""
    out = []
    for i in range(0, len(ch), batch_size):
        x = torch.from_numpy(ch[i:i + batch_size, :-1]).to(device)
        lp_ref = F.log_softmax(ref(x)[0].float(), dim=-1)
        lp_mod = F.log_softmax(model(x)[0].float(), dim=-1)
        kl = (lp_ref.exp() * (lp_ref - lp_mod)).sum(-1)       # (B, T)
        out.append((kl.mean(1) / LN2).cpu().numpy())
    return np.concatenate(out)


def kl_to_reference(ref: GPT, model: GPT, ch: np.ndarray, device: torch.device, batch_size: int = 16) -> float:
    """Mean per-character KL(p_ref || p_model) in bits."""
    return float(kl_per_chunk(ref, model, ch, device, batch_size).mean())


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over chunks (the 1-seed fallback of PLAN §9.5)."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


@torch.no_grad()
def generate_seeded(model: GPT, prompts: list[np.ndarray], seeds: list[int], device: torch.device,
                    max_new: int = 400, temperature: float = 0.8, top_k: int = 40) -> list[np.ndarray]:
    """Generate a continuation for every prompt with its OWN seeded CPU generator.

    Prompts of equal length run as one batch (one forward pass per step), but each row samples
    with its own generator, so a prompt's output does not depend on which others share its batch.
    """
    out: list[np.ndarray | None] = [None] * len(prompts)
    groups = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[len(p)].append(i)
    bs = model.cfg.block_size
    for idxs in groups.values():
        gens = [torch.Generator().manual_seed(seeds[i]) for i in idxs]
        x = torch.from_numpy(np.stack([prompts[i] for i in idxs]).astype(np.int64)).to(device)
        for _ in range(max_new):
            logits = model(x[:, -bs:])[0][:, -1, :].float() / temperature
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            probs = F.softmax(logits.masked_fill(logits < kth, float("-inf")), dim=-1).cpu()
            nxt = torch.stack([torch.multinomial(probs[r], 1, generator=g) for r, g in enumerate(gens)])
            x = torch.cat([x, nxt.to(device)], dim=1)
        for r, i in enumerate(idxs):
            out[i] = x[r, len(prompts[i]):].cpu().numpy()
    return out  # type: ignore[return-value]


def prompts_for(data_dir: Path, lang: str, split: str, n: int = 200) -> list[np.ndarray]:
    vocab = Vocab.load(data_dir / "vocab.json")
    return make_prompts(np.asarray(load_split(data_dir / "processed", split)), vocab, n, PROMPT_SEEDS[lang])


def generation_report(model: GPT, data_dir: Path, wordlists: dict, device: torch.device,
                      prompt_splits: dict[str, str], n: int = 200, max_new: int = 400) -> tuple[dict, dict]:
    """Leakage metrics per prompt language, plus the decoded samples."""
    vocab = Vocab.load(data_dir / "vocab.json")
    metrics, samples = {}, {}
    for lang, split in prompt_splits.items():
        prompts = prompts_for(data_dir, lang, split, n)
        seeds = [100_000 * (1 if lang == "pt" else 2) + i for i in range(len(prompts))]
        gens = generate_seeded(model, prompts, seeds, device, max_new=max_new)
        texts = [vocab.decode(g) for g in gens]
        metrics[f"{lang}_prompts"] = leakage(texts, wordlists)
        samples[lang] = [{"prompt": vocab.decode(p), "generation": t} for p, t in zip(prompts, texts)]
    return metrics, samples
