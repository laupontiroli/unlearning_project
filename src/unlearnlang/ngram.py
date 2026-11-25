"""Count-based character n-gram language model (baseline).

Generalizes the count-based bigram model from Karpathy's *makemore part 1*:
count how often each character follows each context, add `alpha` fake counts
(Laplace smoothing, so no probability is 0 and the NLL never becomes inf),
normalize each row into a distribution, sample with torch.multinomial, and
evaluate with average negative log-likelihood. The article's "." start/end
token is our EOS symbol. order=2 is exactly the article's bigram model.

Because the model is just a table of counts, it also gives an exact-unlearning
illustration: counts are additive, so (PT+EN counts) - (EN counts) == PT counts.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class NGramLM:
    def __init__(self, vocab_size: int, order: int = 2, alpha: float = 1.0):
        assert 1 <= order <= 4, "dense table: order <= 4 keeps V^order manageable"
        self.V, self.order, self.alpha = vocab_size, order, alpha
        self.n_ctx = vocab_size ** (order - 1)
        # counts[context_index, next_char]; context index = base-V number of the previous order-1 chars
        self.counts = np.zeros((self.n_ctx, vocab_size), dtype=np.int64)

    # ---------------------------------------------------------------- counting
    def _grams(self, ids: np.ndarray) -> np.ndarray:
        """Flat index context*V + next for every position with a full context."""
        ids = np.asarray(ids, dtype=np.int64)
        n = len(ids) - self.order + 1
        flat = np.zeros(n, dtype=np.int64)
        for k in range(self.order):
            flat = flat * self.V + ids[k:k + n]
        return flat

    def count(self, ids: np.ndarray, chunk: int = 10_000_000) -> np.ndarray:
        """Count table of one corpus, (n_ctx, V). Chunks overlap by order-1 chars so no gram is lost."""
        total = np.zeros(self.n_ctx * self.V, dtype=np.int64)
        step = chunk
        for s in range(0, max(len(ids) - self.order + 1, 1), step):
            part = ids[s:s + step + self.order - 1]
            if len(part) >= self.order:
                total += np.bincount(self._grams(part), minlength=self.n_ctx * self.V)
        return total.reshape(self.n_ctx, self.V)

    def fit(self, ids: np.ndarray) -> "NGramLM":
        self.counts = self.counts + self.count(ids)
        return self

    def add_counts(self, counts: np.ndarray, sign: int = 1) -> "NGramLM":
        self.counts = self.counts + sign * counts
        assert (self.counts >= 0).all(), "removed counts that were never added"
        return self

    # ------------------------------------------------------------- probability
    def log_probs(self) -> np.ndarray:
        """(n_ctx, V) natural-log probabilities with add-alpha smoothing."""
        c = self.counts.astype(np.float64) + self.alpha
        return np.log(c / c.sum(axis=1, keepdims=True))

    def bpc(self, ids: np.ndarray) -> float:
        """Average NLL per predicted character, in bits (= nats / ln 2)."""
        lp = self.log_probs().reshape(-1)
        nll = -lp[self._grams(ids)].mean()
        return float(nll / math.log(2))

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(self, prompt: np.ndarray, max_new: int, temperature: float = 1.0,
                 top_k: int | None = None, generator: torch.Generator | None = None) -> np.ndarray:
        assert len(prompt) >= self.order - 1, "prompt shorter than the context"
        probs = torch.from_numpy(self.log_probs()).float()
        out = [int(i) for i in prompt]
        for _ in range(max_new):
            ctx = 0
            for i in out[len(out) - (self.order - 1):] if self.order > 1 else []:
                ctx = ctx * self.V + i
            logits = probs[ctx] / temperature
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[-1]
                logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
            p = torch.softmax(logits, dim=-1)
            out.append(int(torch.multinomial(p, 1, replacement=True, generator=generator)))
        return np.array(out, dtype=np.uint8)
