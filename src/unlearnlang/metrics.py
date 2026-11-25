"""Generation prompts and language-leakage metrics (PLAN §9.2). Shared by every model."""
from __future__ import annotations

import numpy as np

from unlearnlang.data import Vocab, extract_words


def make_prompts(ids: np.ndarray, vocab: Vocab, n: int, seed: int, length: int = 64) -> list[np.ndarray]:
    """n prompts of up to `length` chars: start right after a space, cut at the last word boundary."""
    rng = np.random.default_rng(seed)
    space = vocab.stoi[" "]
    prompts: list[np.ndarray] = []
    while len(prompts) < n:
        o = int(rng.integers(0, len(ids) - 4 * length))
        window = np.asarray(ids[o:o + 4 * length])
        if vocab.eos_id in window[: 2 * length]:
            continue
        starts = np.flatnonzero(window == space)
        if len(starts) == 0:
            continue
        w = window[starts[0] + 1: starts[0] + 1 + length]
        cut = np.flatnonzero(w == space)
        if len(cut) == 0:
            continue
        prompts.append(w[: cut[-1] + 1].copy())  # keep the trailing space
    return prompts


def leakage(texts: list[str], wordlists: dict) -> dict:
    """EN-leakage = EN_only / (EN_only + PT_only) generated words; real-word rate = share in REAL."""
    en_only, pt_only, real = (set(wordlists[k]) for k in ("EN_only", "PT_only", "REAL"))
    words = [w for t in texts for w in extract_words(t)]
    n_en = sum(w in en_only for w in words)
    n_pt = sum(w in pt_only for w in words)
    return {
        "en_leakage": n_en / (n_en + n_pt) if n_en + n_pt else float("nan"),
        "real_word_rate": sum(w in real for w in words) / len(words) if words else float("nan"),
        "n_words": len(words), "n_en_only": n_en, "n_pt_only": n_pt,
    }
