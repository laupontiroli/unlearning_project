"""Keeping the Drive working copy honest (see docs/colab_pipeline.md).

From Phase 3 on, MyDrive/unlearn/{checkpoints,results} is the working copy. Two helpers:
- merge_results: bring the Mac's small result files into Drive without ever overwriting Drive data.
- verify_hashes: Drive cannot chmod, so every protected checkpoint is re-hashed at the start and end
  of each notebook and compared with its SHA-256 record.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .utils import sha256_file

HASH_FILES = ("ref_hashes.json", "model_hashes.json")


def merge_results(src: Path, dst: Path) -> list[str]:
    """Copy files from src into dst. Never overwrites: hash records are merged key by key (a key with a
    different SHA raises), JSONL logs get the missing lines appended, any other file that differs is
    reported and left as it is on Drive. Returns a list of what happened."""
    report = []
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = f.relative_to(src)
        g = dst / rel
        if not g.exists():
            g.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, g)
            report.append(f"copied   {rel}")
        elif f.read_bytes() == g.read_bytes():
            continue
        elif f.name in HASH_FILES:
            a = json.loads(g.read_text(encoding="utf-8"))
            b = json.loads(f.read_text(encoding="utf-8"))
            for k, v in b.items():
                if k in a and a[k] != v:
                    raise RuntimeError(f"{rel}: {k} has different SHAs on Mac and Drive; refusing to merge")
            g.write_text(json.dumps({**a, **b}, indent=1), encoding="utf-8")
            report.append(f"merged   {rel} (+{len(set(b) - set(a))} keys)")
        elif f.suffix == ".jsonl":
            have = g.read_text(encoding="utf-8").splitlines()
            new = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln not in set(have)]
            with g.open("a", encoding="utf-8") as fh:
                fh.writelines(ln + "\n" for ln in new)
            report.append(f"appended {rel} (+{len(new)} lines)")
        else:
            report.append(f"KEPT DRIVE VERSION {rel} (differs from the Mac copy)")
    return report


def verify_hashes(results: Path, ckpt_dir: Path) -> list[str]:
    """Re-hash every recorded checkpoint present in ckpt_dir. Raises on any mismatch."""
    report = []
    for name in HASH_FILES:
        rec = results / name
        if not rec.exists():
            report.append(f"no record {name}")
            continue
        for key, want in json.loads(rec.read_text(encoding="utf-8")).items():
            p = ckpt_dir / key
            if not p.exists():
                report.append(f"absent here (not checked): {key}")
                continue
            got = sha256_file(p)
            if got != want:
                raise RuntimeError(f"{key}: SHA-256 {got[:12]}... does not match the record {want[:12]}... (MODIFIED)")
            report.append(f"OK {key} {got[:12]}")
    return report
