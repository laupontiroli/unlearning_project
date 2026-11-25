"""Drive sync helpers must never overwrite Drive data and must catch modified checkpoints."""
from __future__ import annotations

import json

import pytest

from unlearnlang.colab import merge_results, verify_hashes
from unlearnlang.utils import sha256_file


def test_merge_never_overwrites(tmp_path):
    mac, drive = tmp_path / "mac", tmp_path / "drive"
    mac.mkdir(); drive.mkdir()
    (mac / "ref_hashes.json").write_text(json.dumps({"ref_s0.pt": "aa"}), encoding="utf-8")
    (drive / "ref_hashes.json").write_text(json.dumps({"ref_long_s0.pt": "bb"}), encoding="utf-8")
    (mac / "test_access.jsonl").write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    (drive / "test_access.jsonl").write_text('{"b": 2}\n', encoding="utf-8")
    (mac / "notes.txt").write_text("mac", encoding="utf-8")
    (drive / "notes.txt").write_text("drive", encoding="utf-8")
    (mac / "eval.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    merge_results(mac, drive)
    assert json.loads((drive / "ref_hashes.json").read_text()) == {"ref_long_s0.pt": "bb", "ref_s0.pt": "aa"}
    assert (drive / "test_access.jsonl").read_text().splitlines() == ['{"b": 2}', '{"a": 1}']
    assert (drive / "notes.txt").read_text() == "drive"
    assert (drive / "eval.jsonl").exists()
    merge_results(mac, drive)  # idempotent
    assert len((drive / "test_access.jsonl").read_text().splitlines()) == 2


def test_merge_refuses_conflicting_sha(tmp_path):
    mac, drive = tmp_path / "mac", tmp_path / "drive"
    mac.mkdir(); drive.mkdir()
    (mac / "ref_hashes.json").write_text(json.dumps({"ref_long_s0.pt": "aa"}), encoding="utf-8")
    (drive / "ref_hashes.json").write_text(json.dumps({"ref_long_s0.pt": "bb"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        merge_results(mac, drive)


def test_verify_detects_modified_checkpoint(tmp_path):
    ck, res = tmp_path / "ck", tmp_path / "res"
    ck.mkdir(); res.mkdir()
    (ck / "ref_long_s0.pt").write_bytes(b"weights")
    (res / "ref_hashes.json").write_text(json.dumps({"ref_long_s0.pt": sha256_file(ck / "ref_long_s0.pt"),
                                                     "ref_s0.pt": "00"}), encoding="utf-8")
    out = verify_hashes(res, ck)
    assert any(s.startswith("OK ref_long_s0.pt") for s in out)
    assert any("absent here" in s for s in out)
    (ck / "ref_long_s0.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        verify_hashes(res, ck)
