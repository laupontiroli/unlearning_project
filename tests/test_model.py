"""PLAN §5 model tests. These FAIL until Laura implements src/unlearnlang/model.py."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from unlearnlang.model import GPT, ModelConfig, causal_attention_by_hand

V = 62


def small_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=V, block_size=64, n_layer=2, n_head=4, d_model=64, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def test_attention_by_hand_matches_sdpa():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    ours = causal_attention_by_hand(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_output_shape_and_loss(use_sdpa):
    torch.manual_seed(0)
    m = GPT(small_cfg(use_sdpa=use_sdpa))
    x = torch.randint(0, V, (3, 20))
    logits, loss = m(x)
    assert logits.shape == (3, 20, V) and loss is None
    logits, loss = m(x, x)
    assert loss.ndim == 0


def test_sdpa_and_by_hand_paths_agree():
    torch.manual_seed(0)
    a = GPT(small_cfg(use_sdpa=False))
    b = GPT(small_cfg(use_sdpa=True))
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, V, (2, 32))
    torch.testing.assert_close(a(x)[0], b(x)[0], atol=1e-5, rtol=1e-5)


def test_param_count_matches_formula():
    cfg = ModelConfig(vocab_size=63)
    n = GPT(cfg).num_params()
    d = cfg.d_model
    formula = cfg.n_layer * 12 * d * d + cfg.block_size * d + cfg.vocab_size * d
    assert abs(n - formula) / formula < 0.01, (n, formula)


def test_tied_embeddings():
    m = GPT(small_cfg())
    tied = [p for p in m.parameters() if p.shape == (V, 64)]
    assert len(tied) == 1, "output head must share the token-embedding weight"


def test_causal_mask():
    torch.manual_seed(0)
    m = GPT(small_cfg()).eval()
    x = torch.randint(0, V, (1, 40))
    t = 25
    x2 = x.clone()
    x2[0, t] = (x[0, t] + 1) % V
    l1, l2 = m(x)[0], m(x2)[0]
    torch.testing.assert_close(l1[:, :t], l2[:, :t], atol=1e-6, rtol=0)
    assert not torch.allclose(l1[:, t:], l2[:, t:]), "changing position t must affect positions >= t"


def test_initial_loss_near_ln_v():
    torch.manual_seed(0)
    m = GPT(ModelConfig(vocab_size=V, dropout=0.0)).eval()
    x = torch.randint(0, V, (8, 128))
    y = torch.randint(0, V, (8, 128))
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(V)) < 0.1, f"initial loss {loss.item():.3f} vs ln(V) = {math.log(V):.3f}"


def test_overfit_single_batch():
    torch.manual_seed(0)
    m = GPT(small_cfg())
    x = torch.randint(0, V, (4, 64))
    y = torch.randint(0, V, (4, 64))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"could not overfit one batch: loss {loss.item():.3f}"


def test_generate_shape_and_determinism():
    m = GPT(small_cfg()).eval()
    prompt = torch.randint(0, V, (2, 10))
    g1 = m.generate(prompt, 100, temperature=0.8, top_k=10, generator=torch.Generator().manual_seed(1))
    g2 = m.generate(prompt, 100, temperature=0.8, top_k=10, generator=torch.Generator().manual_seed(1))
    assert g1.shape == (2, 110)
    assert torch.equal(g1[:, :10], prompt)
    assert torch.equal(g1, g2)
