"""PLAN §4.6: data tests. Runs on every prepared dataset found (data/ and/or data/smoke/)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from unlearnlang.data import SPLITS, Vocab, build_vocab, clean_text, split_for

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [d for d in (ROOT / "data", ROOT / "data" / "smoke") if (d / "stats.json").exists()]


def test_some_dataset_prepared():
    assert DATA_DIRS, "no prepared dataset: run scripts/prepare_data.py first"


def _titles(d: Path, split: str) -> list[str]:
    lines = (d / "processed" / f"{split}_titles.txt").read_text(encoding="utf-8").split("\n")
    return [t for t in lines if t]


def _bin(d: Path, split: str) -> np.ndarray:
    return np.fromfile(d / "processed" / f"{split}.bin", dtype=np.uint8)


@pytest.fixture(params=DATA_DIRS, ids=lambda p: str(p.relative_to(ROOT)))
def data_dir(request) -> Path:
    return request.param


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_no_title_in_two_splits(data_dir, lang):
    seen: dict[str, str] = {}
    for split in SPLITS[lang]:
        for t in _titles(data_dir, split):
            assert t not in seen, f"{t!r} in both {seen[t]} and {split}"
            seen[t] = split


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_titles_hash_to_their_split(data_dir, lang):
    for split in SPLITS[lang]:
        for t in _titles(data_dir, split):
            assert split_for(lang, t) == split


def test_relearn_fresh_disjoint_from_forget(data_dir):
    fresh, forget = set(_titles(data_dir, "en_relearn_fresh")), set(_titles(data_dir, "en_forget_train"))
    assert fresh and forget and not (fresh & forget)
    # content check: random 64-char windows of relearn_fresh must not occur in en_forget_train
    vocab = Vocab.load(data_dir / "vocab.json")
    fresh_txt = vocab.decode(_bin(data_dir, "en_relearn_fresh"))
    forget_txt = vocab.decode(_bin(data_dir, "en_forget_train"))
    rng = np.random.default_rng(0)
    hits = 0
    for o in rng.integers(0, len(fresh_txt) - 64, size=200):
        w = fresh_txt[o:o + 64]
        if "\n" not in w and w.count(" ") > 5 and w in forget_txt:
            hits += 1
    # a few boilerplate phrases may repeat across articles; real overlap would give many hits
    assert hits <= 5, f"{hits}/200 windows of relearn_fresh found verbatim in en_forget_train"


def test_vocab_round_trip(data_dir):
    vocab = Vocab.load(data_dir / "vocab.json")
    assert vocab.itos == build_vocab().itos
    assert len(vocab) <= 255
    raw = "Ação, coração e pão — “Olá!” São Paulo (1554); the QUICK fox: naïve?"
    text, _ = clean_text(raw)
    assert vocab.decode(vocab.encode(text)) == text
    for split in SPLITS["pt"] + SPLITS["en"]:
        ids = _bin(data_dir, split)[:20000]
        assert (vocab.encode(vocab.decode(ids)) == ids).all()


def test_cleaning_examples():
    text, unk = clean_text("Ação\n\n  É   “bom”\tñ")
    assert text == 'ação é "bom" □'
    assert unk == pytest.approx(1 / len(text))


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_unk_rate_and_eos(data_dir, lang):
    cfg = json.loads((data_dir / "stats.json").read_text(encoding="utf-8"))
    vocab = Vocab.load(data_dir / "vocab.json")
    max_unk = cfg["config"]["max_unk_rate"]
    for split in SPLITS[lang]:
        ids = _bin(data_dir, split)
        assert ids.max() < len(vocab)
        assert (ids == vocab.unk_id).mean() <= max_unk, split
        assert (ids == vocab.eos_id).sum() == len(_titles(data_dir, split)), split
        assert ids[-1] == vocab.eos_id
        assert len(ids) >= cfg["config"]["quotas"][split], f"{split} below quota"


def test_purity_verdict_rules():
    from unlearnlang.data import purity_verdict
    assert purity_verdict("pt", en_share=0.06, pt_share=0.2, max_other=0.05) == "purity_other_share"
    assert purity_verdict("pt", en_share=0.03, pt_share=0.2, max_other=0.05) is None
    assert purity_verdict("en", en_share=0.2, pt_share=0.08, max_other=None) is None
    assert purity_verdict("en", en_share=0.05, pt_share=0.08, max_other=None) == "purity_other_dominant"
