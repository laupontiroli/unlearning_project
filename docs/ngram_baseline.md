# N-gram baseline: the article's model applied to our data

**Source.** Karpathy's *makemore part 1*: a count-based character bigram model trained on 32k names. I followed the write-up "My Notes on Karpathy's Makemore part 1: Building a Bigram Language Model from Scratch" (DEV Community). The method:
1. Count how often each character follows each character, in a matrix N. A "." token marks the start and end of a word.
2. Add 1 to every count (Laplace smoothing). Without it, an unseen pair has probability 0 and the loss becomes infinite.
3. Divide each row by its sum to get probabilities. Generate with `torch.multinomial` and a seeded generator.
4. Measure quality as the average negative log-likelihood. The article reports 2.454 nats per character on the names.

**How it is applied here.** The code is in `src/unlearnlang/ngram.py` and `scripts/ngram_baseline.py`.
- Our EOS symbol plays the role of the article's ".". The vocabulary has 62 symbols.
- The context is generalized from 1 previous character (bigram, order 2) to 3 previous characters (order 4).
- The smoothing amount α is chosen on `pt_val` from {1, 0.1, 0.01, 0.001}; α = 0.1 won at every order. No test data is used.
- Three models per order, mirroring the main experiment: **ref** (PT counts, standing in for θ_ref), **bilingual** (PT + EN counts, standing in for θ_b), and **unlearned_exact** (bilingual counts minus EN counts).
- Generation and leakage follow §9.2 (200 prompts, 400 characters, T = 0.8, top-k 40), but the prompts come from the validation splits.

**Results** (bits per character on the validation splits; uniform guessing = log2 62 = 5.95):

| order | model | PT val | EN val | EN-leakage, PT prompts | real-word rate |
|---|---|---|---|---|---|
| 2 | ref | 3.53 | 4.21 | 0.08 | 0.32 |
| 2 | bilingual | 3.65 | 3.73 | 0.34 | 0.30 |
| 3 | ref | 3.05 | 3.89 | 0.02 | 0.41 |
| 3 | bilingual | 3.19 | 3.24 | 0.31 | 0.44 |
| 4 | ref | 2.56 | 3.61 | 0.008 | 0.52 |
| 4 | bilingual | 2.71 | 2.73 | 0.42 | 0.53 |

For every order, `unlearned_exact` equals `ref` count for count, so it has identical bpc (the test `test_exact_unlearning_by_subtraction` checks this).

**What this gives the project**
1. **A floor for the transformer.** θ_ref must beat 2.56 bpc on PT val, otherwise the transformer adds nothing over counting. The gap between EN and PT for ref (3.61 vs 2.56) is the "shared letters" effect. A model that never saw English still predicts English much better than chance (5.95) because the two languages share letters and many bigrams.
2. **The leakage metric works.** Adding English counts raises EN-leakage on Portuguese prompts from 0.008 to 0.42. The generated text visibly mixes in "the", "and", "of" (`results/ngram_samples.txt`).
3. **Exact vs approximate unlearning, in one line.** For a count model, removing a language is simple subtraction: the counts add up, so you subtract the English ones. A transformer's weights do not decompose by training example, so it needs approximate methods (GA, NegGrad+), and θ_ref is how we measure how far they fall short. This is a good interview answer.
4. **A limitation to mention.** An n-gram sees only the last 1 to 3 characters, so its continuation hardly depends on the prompt. The transformer, with 512 characters of context, should do much better.
