# Phase 1: Model (plain-language summary)

1. `src/unlearnlang/model.py` is a 6-layer, 4-head, 256-wide decoder-only transformer over 62 characters (4.89M parameters). It is adapted from Karpathy's nanoGPT, at Laura's request.
2. Characters become vectors (token embedding) and get a learned position vector. They then pass through 6 blocks, a final LayerNorm and the output head.
3. Each block lets positions exchange information (causal self-attention), then processes each position on its own (MLP). Both steps sit on residual connections with LayerNorm applied first.
4. The causal mask stops a position from seeing later characters, so predicting the next character is never trivial.
5. The output head reuses the embedding matrix (weight tying).
6. Initialization follows GPT-2: N(0, 0.02), with residual output projections scaled by 1/√(2·6). This gives an initial loss ≈ ln 62.
7. The loss is the mean cross-entropy in nats over every position; bpc = loss / ln 2.
8. `generate` samples one character at a time with temperature, top-k and a seeded generator.
9. Tests (`tests/test_model.py`, all 10 passing) cover: attention by hand vs SDPA, both paths giving the same logits, shapes, parameter count, tying, the causal mask, initial loss ≈ ln V, overfitting one batch, and deterministic generation.
10. Gate 1 numbers are in `results/gate1.json`. The line-by-line explanation is in `docs/model_walkthrough.md`.
