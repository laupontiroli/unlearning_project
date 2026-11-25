"""Gate 0 report: stats table, snippets, word lists, dropped articles, purity histogram.

Usage: uv run python scripts/inspect_data.py --data-dir data [--n-dropped 20]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unlearnlang.data import SPLITS, Vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-dropped", type=int, default=20)
    args = ap.parse_args()
    d = Path(args.data_dir)
    stats = json.loads((d / "stats.json").read_text(encoding="utf-8"))
    vocab = Vocab.load(d / "vocab.json")
    rng = np.random.default_rng(0)

    print(f"vocab size: {stats['vocab_size']}\n")
    print(f"{'split':18s} {'chars':>12s} {'articles':>9s} {'chars/art':>9s} {'unk':>7s}")
    for lang in ("pt", "en"):
        for s, v in stats[lang]["splits"].items():
            print(f"{s:18s} {v['characters']:12,d} {v['articles']:9,d} {v['characters'] / max(v['articles'], 1):9,.0f} {v['unk_rate']:7.3%}")
    for lang in ("pt", "en"):
        st = stats[lang]
        print(f"\n[{lang}] rows read {st['rows_read']:,} in {st['seconds']:.0f}s; row groups {st['row_groups_read']}/{st['row_groups_total']}"
              f"; per dump decile {st['row_groups_per_dump_decile']}")
        print(f"[{lang}] dropped: {st['dropped']}")

    for split in ("pt_train", "en_forget_train"):
        ids = np.fromfile(d / "processed" / f"{split}.bin", dtype=np.uint8)
        print(f"\n== 5 random snippets of {split}")
        for o in rng.integers(0, len(ids) - 200, 5):
            print("  |" + vocab.decode(ids[o:o + 180]).replace("\n", "⏎") + "|")

    wl = json.loads((d / "wordlists.json").read_text(encoding="utf-8"))
    print(f"\nword lists: {wl['sizes']}  (corpus words: {wl['corpus_words']})")
    for k in ("EN_only", "PT_only"):
        print(f"top {k}: " + ", ".join(f"{w} ({f})" for w, f in wl["top_freq_per_million"][k][:10]))

    for lang in ("pt", "en"):
        items = [json.loads(l) for l in (d / "inspect" / f"dropped_{lang}.jsonl").read_text(encoding="utf-8").splitlines() if l]
        pick = random.Random(0).sample(items, min(args.n_dropped, len(items)))
        print(f"\n== {len(pick)} random purity-dropped {lang.upper()} articles (of {len(items)} in reservoir)")
        for it in pick:
            print(f"  [{it['reason']}, en={it['en_share']:.3f} pt={it['pt_share']:.3f}, {it['n_tokens']} tok] {it['title']}")
            print(f"     {it['snippet'][:160]}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, lang in zip(axes, ("pt", "en")):
        z = np.load(d / "inspect" / f"purity_{lang}.npz", allow_pickle=True)
        other = z["en_share"] if lang == "pt" else z["pt_share"]
        ax.hist(other, bins=np.linspace(0, 0.15, 61), log=True)
        thr = stats["config"]["purity"]["max_other_share"]
        thr = thr.get(lang) if isinstance(thr, dict) else thr
        if thr is not None:
            ax.axvline(thr, color="k", ls="--", lw=1)
        ax.set_title(f"{lang.upper()} articles: share of {'EN' if lang == 'pt' else 'PT'} stopwords")
        ax.set_xlabel("share of tokens"); ax.set_ylabel("articles (log)")
    fig.tight_layout()
    out = Path("results/figures/purity_hist.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
