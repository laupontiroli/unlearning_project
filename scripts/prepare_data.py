"""PHASE 0: sample Wikipedia (random row groups), clean, filter, split by article hash, pack to uint8 .bin files.

Usage:
    uv run python scripts/prepare_data.py --config configs/data_smoke.yaml
    uv run python scripts/prepare_data.py --config configs/data.yaml
    uv run python scripts/prepare_data.py --config configs/data.yaml --wordlists-only

Outputs (in cfg.out_dir):
    vocab.json, stats.json, wordlists.json
    processed/{split}.bin, processed/{split}_titles.txt
    inspect/purity_{lang}.npz          stopword shares of every article that reached the purity check
    inspect/dropped_{lang}.jsonl       reservoir sample of purity-dropped articles (title, shares, snippet)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

import numpy as np
import yaml
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from unlearnlang.data import (
    SPLITS, Vocab, build_vocab, clean_text, extract_words, keep_article, purity_verdict,
    split_for, stopword_shares,
)

RESERVOIR = 300
PREFETCH = 8  # row groups fetched in parallel


def row_group_index(cfg: dict, lang: str) -> list[tuple[str, int, int]]:
    """All (shard_path, row_group, global_position) of one dump, in dump order (cached on disk)."""
    cache = Path("data/raw") / f"rowgroups_{cfg['configs'][lang]}.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text(encoding="utf-8"))]
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{cfg['hf_dataset']}/{cfg['configs'][lang]}/*.parquet"))
    index = []
    for sh in shards:
        with fs.open(sh, "rb") as f:
            for rg in range(pq.ParquetFile(f).num_row_groups):
                index.append((sh, rg, len(index)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index), encoding="utf-8")
    return index


def _read_row_group(item: tuple[str, int, int]):
    """One row group's (title, text) table, cached in data/raw/rowgroups/."""
    sh, rg, _ = item
    cache = Path("data/raw/rowgroups") / f"{sh.split('/')[-2]}__{Path(sh).stem}__{rg:04d}.parquet"
    if cache.exists():
        return pq.read_table(cache)
    with HfFileSystem().open(sh, "rb") as f:
        tab = pq.ParquetFile(f).read_row_group(rg, columns=["title", "text"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tab, cache)
    return tab


def iter_articles(cfg: dict, lang: str, used: list[int]):
    """Yield (title, text) from row groups visited in a seeded random order.

    Row groups (~1000 consecutive articles each) are a random sample across the *whole*
    dump, and because they are visited in random order every split fills uniformly over
    the dump instead of from its beginning (DEVIATION from PLAN §4.1, see results/deviations.md).
    """
    index = row_group_index(cfg, lang)
    order = list(index)
    random.Random(f"rowgroups-{lang}-{cfg['sampling_seed']}").shuffle(order)
    with ThreadPoolExecutor(PREFETCH) as ex:
        for start in range(0, len(order), PREFETCH):
            batch = order[start:start + PREFETCH]
            for item, tab in zip(batch, ex.map(_read_row_group, batch)):
                used.append(item[2])
                for title, text in zip(tab.column("title").to_pylist(), tab.column("text").to_pylist()):
                    yield title, text


def process_language(lang: str, cfg: dict, vocab: Vocab, out: Path) -> dict:
    quotas = {s: int(cfg["quotas"][s]) for s in SPLITS[lang]}
    texts: dict[str, list[str]] = {s: [] for s in quotas}
    titles: dict[str, list[str]] = {s: [] for s in quotas}
    chars = Counter()
    drops = Counter()
    en_set = frozenset(cfg["purity"]["en_stopwords"])
    pt_set = frozenset(cfg["purity"]["pt_stopwords"])
    max_other = cfg["purity"]["max_other_share"][lang]  # None = only the "other dominates" rule
    shares, share_titles, reservoir, n_purity_drops = [], [], [], 0
    rng = random.Random(f"reservoir-{lang}")

    t0, n_seen, used = time.time(), 0, []
    for title, raw in iter_articles(cfg, lang, used):
        n_seen += 1
        if n_seen % 20000 == 0:
            filled = ", ".join(f"{s}={chars[s] / quotas[s]:.0%}" for s in quotas)
            print(f"[{lang}] {n_seen:,} rows, {len(used)} row groups, {time.time() - t0:.0f}s | {filled}", flush=True)
        if not keep_article(title, cfg["keep_modulus"]):
            continue
        split = split_for(lang, title)
        if split is None:
            drops["bucket_ignored"] += 1
            continue
        if chars[split] >= quotas[split]:
            drops["quota_full"] += 1
            continue
        text, unk_rate = clean_text(raw, cfg["normalize_typography"])
        if not text:
            drops["empty"] += 1
            continue
        if unk_rate > cfg["max_unk_rate"]:
            drops["unk_rate"] += 1
            continue
        en_s, pt_s, n_tok = stopword_shares(text, en_set, pt_set)
        shares.append((en_s, pt_s, n_tok))
        share_titles.append(title)
        reason = purity_verdict(lang, en_s, pt_s, max_other)
        if reason:
            drops[reason] += 1
            n_purity_drops += 1
            item = {"title": title, "split": split, "en_share": round(en_s, 4), "pt_share": round(pt_s, 4),
                    "n_tokens": n_tok, "reason": reason, "snippet": text[:400]}
            if len(reservoir) < RESERVOIR:
                reservoir.append(item)
            else:
                j = rng.randrange(n_purity_drops)
                if j < RESERVOIR:
                    reservoir[j] = item
            continue
        texts[split].append(text)
        titles[split].append(title)
        chars[split] += len(text) + 1  # +1 for EOS
        if all(chars[s] >= quotas[s] for s in quotas):
            break
    else:
        print(f"[{lang}] WARNING: dataset exhausted before all quotas filled", flush=True)

    elapsed = time.time() - t0
    n_rg = len(row_group_index(cfg, lang))
    stats = {"rows_read": n_seen, "seconds": round(elapsed, 1), "dropped": dict(drops), "splits": {},
             "row_groups_read": len(used), "row_groups_total": n_rg,
             "row_groups_per_dump_decile": np.histogram(used, bins=10, range=(0, n_rg))[0].tolist()}
    proc = out / "processed"
    for s in quotas:
        joined = "".join(t + vocab.itos[vocab.eos_id] for t in texts[s])
        ids = vocab.encode(joined)
        ids.tofile(proc / f"{s}.bin")
        (proc / f"{s}_titles.txt").write_text("\n".join(titles[s]) + "\n", encoding="utf-8")
        stats["splits"][s] = {
            "characters": int(len(ids)),
            "quota": quotas[s],
            "articles": len(texts[s]),
            "unk_rate": float((ids == vocab.unk_id).mean()) if len(ids) else None,
        }
        print(f"[{lang}] {s}: {len(ids):,} chars, {len(texts[s]):,} articles", flush=True)

    insp = out / "inspect"
    arr = np.array(shares, dtype=np.float64).reshape(-1, 3)
    np.savez(insp / f"purity_{lang}.npz", en_share=arr[:, 0], pt_share=arr[:, 1], n_tokens=arr[:, 2],
             title=np.array(share_titles, dtype=object))
    with open(insp / f"dropped_{lang}.jsonl", "w", encoding="utf-8") as f:
        for item in reservoir:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return stats


def word_freqs(ids: np.ndarray, vocab: Vocab) -> tuple[Counter, int]:
    lut = np.array(vocab.itos)
    text = "".join(lut[ids].tolist())
    words = extract_words(text)
    return Counter(words), len(words)


def build_wordlists(cfg: dict, vocab: Vocab, out: Path) -> dict:
    proc = out / "processed"
    c_pt, n_pt = word_freqs(np.fromfile(proc / "pt_train.bin", dtype=np.uint8), vocab)
    c_en, n_en = word_freqs(np.fromfile(proc / "en_forget_train.bin", dtype=np.uint8), vocab)
    f_pt = {w: c / n_pt * 1e6 for w, c in c_pt.items()}
    f_en = {w: c / n_en * 1e6 for w, c in c_en.items()}
    wl = cfg["wordlists"]
    mf, r, real_min = wl["min_freq_per_million"], wl["max_ratio_other"], wl["real_min_freq"]
    en_only = sorted((w for w, f in f_en.items() if f >= mf and f_pt.get(w, 0.0) <= r * f), key=lambda w: -f_en[w])
    pt_only = sorted((w for w, f in f_pt.items() if f >= mf and f_en.get(w, 0.0) <= r * f), key=lambda w: -f_pt[w])
    real = sorted(w for w in set(f_pt) | set(f_en) if max(f_pt.get(w, 0.0), f_en.get(w, 0.0)) >= real_min)
    result = {
        "params": wl,
        "corpus_words": {"pt_train": n_pt, "en_forget_train": n_en},
        "sizes": {"EN_only": len(en_only), "PT_only": len(pt_only), "REAL": len(real)},
        "EN_only": en_only, "PT_only": pt_only, "REAL": real,
        "top_freq_per_million": {
            "EN_only": [[w, round(f_en[w], 1)] for w in en_only[:30]],
            "PT_only": [[w, round(f_pt[w], 1)] for w in pt_only[:30]],
        },
    }
    (out / "wordlists.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wordlists: {result['sizes']}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--langs", nargs="+", default=["pt", "en"])
    ap.add_argument("--wordlists-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg["out_dir"])
    (out / "processed").mkdir(parents=True, exist_ok=True)
    (out / "inspect").mkdir(parents=True, exist_ok=True)

    vocab = build_vocab()
    vocab.save(out / "vocab.json")
    print(f"vocab size {len(vocab)}", flush=True)

    stats_path = out / "stats.json"
    if not args.wordlists_only:
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        stats["config"] = cfg
        stats["vocab_size"] = len(vocab)
        for lang in args.langs:
            stats[lang] = process_language(lang, cfg, vocab, out)
            stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    build_wordlists(cfg, vocab, out)


if __name__ == "__main__":
    main()
