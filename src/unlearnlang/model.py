"""Decoder-only character-level transformer (PLAN §5).

Based on Andrej Karpathy's nanoGPT `model.py` (https://github.com/karpathy/nanoGPT, MIT License,
Copyright (c) 2022 Andrej Karpathy), adapted to this project:
  * a stand-alone `causal_attention_by_hand` (explicit matmul + mask + softmax), checked against SDPA;
  * `use_sdpa` switch instead of auto-detecting flash attention;
  * forward() returns logits for ALL positions (nanoGPT returns only the last one at inference),
    because evaluation needs per-position distributions (bpc, KL to the reference);
  * generate() takes a torch.Generator for reproducible sampling (PLAN §9.2);
  * submodules are named `qkv` / `fc` / `proj` instead of `c_attn` / `c_fc` / `c_proj`.
A line-by-line walkthrough is in docs/model_walkthrough.md.

Notation: B = batch size, T = sequence length (<= block_size), C = d_model,
H = n_head, D = d_head = C // H, V = vocab_size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 62
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1
    use_sdpa: bool = True  # False -> use causal_attention_by_hand


def causal_attention_by_hand(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             dropout_p: float = 0.0, training: bool = False) -> torch.Tensor:
    """Causal scaled dot-product attention with explicit matmul, mask and softmax.

    Args:
        q, k, v: (B, H, T, D)
        dropout_p: dropout on the attention weights (only applied when training=True)
    Returns:
        (B, H, T, D) where output[t] = sum_{s <= t} softmax_s(q_t . k_s / sqrt(D)) v_s
    """
    T, D = q.size(-2), q.size(-1)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(D)                       # (B, H, T, T) similarity scores
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()     # True where s <= t
    att = att.masked_fill(~mask, float("-inf"))                           # future positions -> -inf
    att = F.softmax(att, dim=-1)                                          # each row sums to 1; -inf -> 0
    att = F.dropout(att, p=dropout_p, training=training)
    return att @ v                                                        # (B, H, T, T) @ (B, H, T, D)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.use_sdpa = cfg.use_sdpa
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)   # Q, K, V for all heads in one matmul
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)      # output projection (residual branch)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C). Split into heads (B, H, T, D), attend causally, merge heads."""
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)                                 # each (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)      # (B, H, T, D)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        p = self.dropout if self.training else 0.0
        if self.use_sdpa:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        else:
            y = causal_attention_by_hand(q, k, v, dropout_p=p, training=self.training)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                     # merge heads side by side
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Linear(C, 4C) -> GELU -> Linear(4C, C) -> dropout."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)"""
        return self.dropout(self.proj(self.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block:  x = x + attn(ln1(x));  x = x + mlp(ln2(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)"""
        x = x + self.attn(self.ln1(x))   # tokens exchange information
        x = x + self.mlp(self.ln2(x))    # each position computes on its own
        return x


class GPT(nn.Module):
    """Token embedding (V, C) + learned position embedding (block_size, C) -> dropout
    -> n_layer Blocks -> final LayerNorm -> output head Linear(C, V, bias=False)
    whose weight IS the token-embedding weight (tied).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying: one (V, C) matrix used twice

        # init: N(0, 0.02) everywhere, biases 0; residual output projections scaled down
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     (B, T) int64 character ids, T <= block_size
            targets: (B, T) int64 next-character ids, or None
        Returns:
            logits: (B, T, V)
            loss:   scalar mean cross-entropy in nats over all B*T positions, or None
        """
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)                      # (T,)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))          # (B, T, C) + (T, C) broadcast
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                              # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self) -> int:
        """Number of unique parameters (the tied head counts once)."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        """Autoregressive sampling.

        Args:
            idx: (B, T0) prompt ids
            max_new_tokens: number of characters to append
            temperature: divide the last-position logits by this before softmax
            top_k: if set, keep only the k largest logits (others -> -inf)
            generator: torch.Generator for reproducible sampling (must live on idx's device;
                       pass a CPU generator for CPU sampling)
        Returns:
            (B, T0 + max_new_tokens). The context is cropped to the last block_size ids each step.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature                   # (B, V) next-character scores
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
