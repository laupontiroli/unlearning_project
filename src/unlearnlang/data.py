"""Data pipeline: cleaning, language-purity filter, hash-based splitting, packing, batch sampler.

See PLAN §4. All text I/O uses explicit UTF-8.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Vocabulary (PLAN §4.2)
# ---------------------------------------------------------------------------
LETTERS = "abcdefghijklmnopqrstuvwxyz"
ACCENTED = "áàâãçéêíóôõú"
DIGITS = "0123456789"
PUNCT = " .,;:!?-()'\""
ALLOWED = LETTERS + ACCENTED + DIGITS + PUNCT  # 60 characters

# Display characters for the two special symbols. Neither can survive cleaning as a
# normal character ("\n" is collapsed to a space, "□" is not in ALLOWED), so they are unambiguous.
UNK_CHAR = "□"
EOS_CHAR = "\n"

# Typographic variants mapped to their ASCII equivalent *before* the UNK check
# (only when cfg normalize_typography is true). They carry the same meaning, so
# counting them as UNK would drop many otherwise clean articles.
TYPO_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
})

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Vocab:
    itos: list[str]

    @property
    def stoi(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.itos)}

    @property
    def unk_id(self) -> int:
        return self.itos.index(UNK_CHAR)

    @property
    def eos_id(self) -> int:
        return self.itos.index(EOS_CHAR)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> np.ndarray:
        """Cleaned text (only ALLOWED + UNK_CHAR + EOS_CHAR) -> uint8 ids."""
        stoi = self.stoi
        return np.fromiter((stoi[c] for c in text), dtype=np.uint8, count=len(text))

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: Path) -> None:
        payload = {"itos": self.itos, "unk_char": UNK_CHAR, "eos_char": EOS_CHAR,
                   "unk_id": self.unk_id, "eos_id": self.eos_id, "size": len(self)}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["itos"])


def build_vocab() -> Vocab:
    v = Vocab(list(ALLOWED) + [UNK_CHAR, EOS_CHAR])
    assert len(set(v.itos)) == len(v.itos), "duplicate characters in vocab"
    assert len(v) <= 255, "vocab must fit in uint8"
    return v


# ---------------------------------------------------------------------------
# Cleaning (PLAN §4.2)
# ---------------------------------------------------------------------------
_ALLOWED_SET = frozenset(ALLOWED)


def clean_text(raw: str, normalize_typography: bool = True) -> tuple[str, float]:
    """NFC -> lowercase -> (typography map) -> collapse whitespace -> non-allowed chars to UNK.

    Returns (cleaned_text, unk_rate). The cleaned text contains no EOS; articles are
    joined with EOS when packed.
    """
    text = unicodedata.normalize("NFC", raw).lower()
    # lowercasing can produce decomposed sequences (e.g. "İ" -> "i̇"); re-normalize.
    text = unicodedata.normalize("NFC", text)
    if normalize_typography:
        text = text.translate(TYPO_MAP)
    text = _WS.sub(" ", text).strip()
    if not text:
        return "", 1.0
    out = [c if c in _ALLOWED_SET else UNK_CHAR for c in text]
    n_unk = sum(1 for c in out if c == UNK_CHAR)
    return "".join(out), n_unk / len(out)


# ---------------------------------------------------------------------------
# Language purity (PLAN §4.3)
# ---------------------------------------------------------------------------
_STRIP = PUNCT.replace(" ", "") + UNK_CHAR


def stopword_shares(text: str, en_set: frozenset[str], pt_set: frozenset[str]) -> tuple[float, float, int]:
    """Share of whitespace tokens (punctuation stripped) in the EN and PT stopword sets."""
    toks = [t.strip(_STRIP) for t in text.split(" ")]
    toks = [t for t in toks if t]
    n = len(toks)
    if n == 0:
        return 0.0, 0.0, 0
    en = sum(1 for t in toks if t in en_set)
    pt = sum(1 for t in toks if t in pt_set)
    return en / n, pt / n, n


def purity_verdict(lang: str, en_share: float, pt_share: float, max_other: float | None) -> str | None:
    """None if the article is kept, otherwise the reason it is dropped.

    max_other=None disables the absolute threshold (only the "other language dominates" rule applies).
    """
    own, other = (pt_share, en_share) if lang == "pt" else (en_share, pt_share)
    if max_other is not None and other > max_other:
        return "purity_other_share"
    if other > own:
        return "purity_other_dominant"
    return None


# ---------------------------------------------------------------------------
# Hash-based sampling and splitting (PLAN §4.1)
# ---------------------------------------------------------------------------
def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)


def keep_article(title: str, modulus: int = 5) -> bool:
    return _h("keep" + title) % modulus == 0


def split_bucket(title: str) -> int:
    return _h("split" + title) % 1000


def split_for(lang: str, title: str) -> str | None:
    """Split name for an article, or None if this language ignores the bucket."""
    b = split_bucket(title)
    if b < 850:
        return "pt_train" if lang == "pt" else "en_forget_train"
    if b < 900:
        return f"{lang}_val"
    if b < 950:
        return f"{lang}_test"
    return "en_relearn_fresh" if lang == "en" else None


SPLITS = {
    "pt": ["pt_train", "pt_val", "pt_test"],
    "en": ["en_forget_train", "en_val", "en_test", "en_relearn_fresh"],
}

# ---------------------------------------------------------------------------
# Word extraction (PLAN §4.5, §9.2)
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r"[a-záàâãçéêíóôõú]{3,}")


def extract_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


# ---------------------------------------------------------------------------
# Batch sampler (PLAN §4.4)
# ---------------------------------------------------------------------------
def load_split(processed_dir: Path, split: str) -> np.ndarray:
    return np.memmap(Path(processed_dir) / f"{split}.bin", dtype=np.uint8, mode="r")


class BatchSampler:
    """Random windows of block_size+1 characters -> (x, y) with y shifted by one.

    Seeded with its own torch.Generator so batches are reproducible and independent
    of any other randomness in the program.
    """

    def __init__(self, data: np.ndarray, block_size: int, batch_size: int, seed: int, device: str | torch.device = "cpu"):
        assert len(data) > block_size + 1, "split shorter than one block"
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.gen = torch.Generator().manual_seed(seed)

    def offsets(self) -> torch.Tensor:
        return torch.randint(0, len(self.data) - self.block_size - 1, (self.batch_size,), generator=self.gen)

    def batch_at(self, offsets) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.block_size
        chunk = np.stack([np.asarray(self.data[o:o + T + 1]) for o in np.asarray(offsets)]).astype(np.int64)
        t = torch.from_numpy(chunk)
        x, y = t[:, :-1], t[:, 1:]
        if self.device.type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.batch_at(self.offsets())

    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, sd: dict) -> None:
        self.gen.set_state(sd["gen"])


def fixed_offsets(n_data: int, block_size: int, n: int, seed: int) -> np.ndarray:
    """Deterministic chunk offsets for evaluation (same chunks every time)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_data - block_size - 1, size=n)
