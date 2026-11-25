"""Count-based character n-gram baseline (makemore part 1, generalized to order 2-4).

For each order:
  * pick the smoothing alpha on pt_val (validation only, PLAN §11.5)
  * ref       = counts on pt_train                (analogue of θ_ref)
  * bilingual = counts on pt_train + en_forget_train  (analogue of θ_b, 50/50 data)
  * unlearned = bilingual counts - en_forget_train counts  (EXACT unlearning; must equal ref)
  * bpc on pt_val / en_val, and generations from 200 pt_val + 200 en_val prompts (§9.2 protocol,
    but on validation prompts) scored for EN-leakage and real-word rate.

Usage: uv run python scripts/ngram_baseline.py [--data-dir data] [--orders 2 3 4]
Output: results/ngram_baseline.json, results/ngram_samples.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

import numpy as np
import torch

from unlearnlang.data import Vocab, load_split
from unlearnlang.metrics import leakage, make_prompts
from unlearnlang.ngram import NGramLM

ALPHAS = [1.0, 0.1, 0.01, 0.001]


def gen_texts(m: NGramLM, prompts, vocab: Vocab, max_new: int, base_seed: int) -> list[str]:
    out = []
    for i, p in enumerate(prompts):
        g = torch.Generator().manual_seed(base_seed + i)
        ids = m.generate(p, max_new, temperature=0.8, top_k=40, generator=g)
        out.append(vocab.decode(ids[len(p):]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--orders", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=400)
    args = ap.parse_args()
    d = Path(args.data_dir)
    proc = d / "processed"
    vocab = Vocab.load(d / "vocab.json")
    wl = json.loads((d / "wordlists.json").read_text(encoding="utf-8"))
    V = len(vocab)
    pt_train, en_train = np.asarray(load_split(proc, "pt_train")), np.asarray(load_split(proc, "en_forget_train"))
    pt_val, en_val = np.asarray(load_split(proc, "pt_val")), np.asarray(load_split(proc, "en_val"))
    prompts = {"pt": make_prompts(pt_val, vocab, args.n_prompts, seed=1),
               "en": make_prompts(en_val, vocab, args.n_prompts, seed=2)}

    results = {"uniform_bpc": float(np.log2(V)), "orders": {}}
    samples = []
    for order in args.orders:
        t0 = time.time()
        probe = NGramLM(V, order)
        c_pt, c_en = probe.count(pt_train), probe.count(en_train)

        # smoothing chosen on pt_val for the PT-only model, then reused for every model of this order
        alpha_bpc = {a: NGramLM(V, order, a).add_counts(c_pt).bpc(pt_val) for a in ALPHAS}
        alpha = min(alpha_bpc, key=alpha_bpc.get)

        ref = NGramLM(V, order, alpha).add_counts(c_pt)
        bil = NGramLM(V, order, alpha).add_counts(c_pt).add_counts(c_en)
        unl = NGramLM(V, order, alpha).add_counts(c_pt).add_counts(c_en).add_counts(c_en, sign=-1)
        exact = bool(np.array_equal(unl.counts, ref.counts))

        r = {"alpha": alpha, "alpha_search_pt_val_bpc": {str(a): round(b, 4) for a, b in alpha_bpc.items()},
             "exact_unlearning_equals_ref": exact, "models": {}}
        for name, m in (("ref", ref), ("bilingual", bil), ("unlearned_exact", unl)):
            row = {"pt_val_bpc": round(m.bpc(pt_val), 4), "en_val_bpc": round(m.bpc(en_val), 4)}
            if name != "unlearned_exact":  # identical to ref when exact == True
                for lang in ("pt", "en"):
                    # one fixed seed per prompt (§9.2); PT and EN prompts get different seeds
                    seed0 = 10_000 * order + (0 if lang == "pt" else 5_000)
                    texts = gen_texts(m, prompts[lang], vocab, args.max_new, base_seed=seed0)
                    row[f"gen_{lang}_prompts"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in leakage(texts, wl).items()}
                    for i in range(3):
                        samples.append(f"[order {order} | {name} | {lang} prompt]\n"
                                       f"PROMPT: {vocab.decode(prompts[lang][i])}\nGEN:    {texts[i][:240]}\n")
            r["models"][name] = row
        r["seconds"] = round(time.time() - t0, 1)
        results["orders"][str(order)] = r
        print(json.dumps({order: r}, ensure_ascii=False, indent=1), flush=True)

    Path("results").mkdir(exist_ok=True)
    Path("results/ngram_baseline.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    Path("results/ngram_samples.txt").write_text("\n".join(samples), encoding="utf-8")
    print("saved results/ngram_baseline.json and results/ngram_samples.txt")


if __name__ == "__main__":
    main()
