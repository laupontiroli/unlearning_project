"""Tests for the count-based n-gram baseline and the leakage metric."""
from __future__ import annotations

import math

import numpy as np
import torch

from unlearnlang.data import build_vocab
from unlearnlang.metrics import leakage, make_prompts
from unlearnlang.ngram import NGramLM


def test_bigram_counts_match_manual():
    ids = np.array([0, 1, 1, 2, 0, 1], dtype=np.uint8)
    m = NGramLM(3, order=2, alpha=0).fit(ids)
    expected = np.zeros((3, 3), dtype=np.int64)
    for a, b in zip(ids[:-1], ids[1:]):
        expected[a, b] += 1
    assert np.array_equal(m.counts, expected)


def test_chunked_counting_loses_no_gram():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 5, size=1000).astype(np.uint8)
    m = NGramLM(5, order=3)
    assert np.array_equal(m.count(ids, chunk=37), m.count(ids, chunk=10_000))
    assert m.count(ids).sum() == len(ids) - 2


def test_bpc_matches_manual_and_uniform():
    ids = np.array([0, 1, 1, 2, 0, 1], dtype=np.uint8)
    m = NGramLM(3, order=2, alpha=1.0).fit(ids)
    c = m.counts + 1.0
    p = c / c.sum(1, keepdims=True)
    manual = -np.mean([math.log2(p[a, b]) for a, b in zip(ids[:-1], ids[1:])])
    assert abs(m.bpc(ids) - manual) < 1e-9
    assert abs(NGramLM(7, order=3, alpha=1.0).bpc(ids) - math.log2(7)) < 1e-9  # no counts -> uniform


def test_exact_unlearning_by_subtraction():
    rng = np.random.default_rng(1)
    a, b = (rng.integers(0, 6, size=500).astype(np.uint8) for _ in range(2))
    ref = NGramLM(6, order=3).fit(a)
    unl = NGramLM(6, order=3).fit(a).fit(b).add_counts(NGramLM(6, order=3).count(b), sign=-1)
    assert np.array_equal(ref.counts, unl.counts)


def test_generate_deterministic_and_keeps_prompt():
    rng = np.random.default_rng(2)
    m = NGramLM(6, order=3).fit(rng.integers(0, 6, size=2000).astype(np.uint8))
    prompt = np.array([1, 2, 3], dtype=np.uint8)
    g1 = m.generate(prompt, 50, 0.8, 4, torch.Generator().manual_seed(0))
    g2 = m.generate(prompt, 50, 0.8, 4, torch.Generator().manual_seed(0))
    assert len(g1) == 53 and np.array_equal(g1[:3], prompt) and np.array_equal(g1, g2)


def test_leakage_counts():
    wl = {"EN_only": ["the", "and"], "PT_only": ["que", "uma"], "REAL": ["the", "and", "que", "uma", "casa"]}
    r = leakage(["the casa que xyzq", "and uma uma"], wl)
    assert r["n_en_only"] == 2 and r["n_pt_only"] == 3
    assert abs(r["en_leakage"] - 2 / 5) < 1e-12
    assert abs(r["real_word_rate"] - 6 / 7) < 1e-12


def test_make_prompts_end_on_word_boundary():
    v = build_vocab()
    text = ("o gato subiu no telhado e a casa ficou em silêncio durante a noite inteira " * 20)
    ids = v.encode(text)
    ps = make_prompts(ids, v, n=5, seed=0)
    for p in ps:
        s = v.decode(p)
        assert len(s) <= 64 and s.endswith(" ") and not s.startswith(" ")
