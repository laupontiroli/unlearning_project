"""Evaluation helpers: per-prompt seeding, KL, bpc."""
from __future__ import annotations

import math

import numpy as np
import torch

from unlearnlang.evaluate import bpc, generate_seeded, kl_to_reference
from unlearnlang.model import GPT, ModelConfig

CPU = torch.device("cpu")


def tiny() -> GPT:
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=20, block_size=32, n_layer=1, n_head=2, d_model=16, dropout=0.0)).eval()


def test_generation_independent_of_batch_grouping():
    m = tiny()
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, 20, size=n) for n in (5, 5, 5, 7)]
    seeds = [10, 11, 12, 13]
    all_out = generate_seeded(m, prompts, seeds, CPU, max_new=40)
    alone = generate_seeded(m, [prompts[1]], [11], CPU, max_new=40)
    assert np.array_equal(all_out[1], alone[0])
    assert len(all_out[3]) == 40
    assert not np.array_equal(all_out[0], all_out[1])  # different seeds/prompts


def test_kl_zero_to_itself_and_positive_otherwise():
    m = tiny()
    ch = np.random.default_rng(1).integers(0, 20, size=(4, 17))
    assert abs(kl_to_reference(m, m, ch, CPU)) < 1e-6
    torch.manual_seed(5)
    other = GPT(ModelConfig(vocab_size=20, block_size=32, n_layer=1, n_head=2, d_model=16, dropout=0.0)).eval()
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.5 * torch.randn_like(p))
    assert kl_to_reference(m, other, ch, CPU) > 0


def test_bpc_of_untrained_model_near_log2_v():
    m = tiny()
    ch = np.random.default_rng(2).integers(0, 20, size=(8, 33))
    assert abs(bpc(m, ch, CPU) - math.log2(20)) < 0.2
