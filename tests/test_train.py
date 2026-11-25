"""Training-loop tests on the smoke data (CPU, tiny model)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from unlearnlang.train import load_config, lr_at, train
from unlearnlang.utils import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "smoke" / "stats.json"


def tiny_cfg(**kw) -> dict:
    cfg = load_config(ROOT / "configs" / "stage1_smoke.yaml")
    cfg["data_dir"] = str(ROOT / "data" / "smoke")
    cfg["model"].update(n_layer=2, n_head=2, d_model=32, block_size=32)
    cfg.update(block_size=32, max_steps=12, eval_interval=4, ckpt_interval=4, eval_batches=2, eval_batch_size=4,
               mixture={"pt_train": 4})
    cfg.update(kw)
    return cfg


def test_lr_schedule():
    cfg = {"lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 10, "max_steps": 110}
    assert lr_at(0, cfg) == pytest.approx(1e-4)
    assert lr_at(9, cfg) == pytest.approx(1e-3)
    assert lr_at(10, cfg) == pytest.approx(1e-3)
    assert lr_at(60, cfg) == pytest.approx(5.5e-4)
    assert lr_at(110, cfg) == pytest.approx(1e-4)


@pytest.mark.skipif(not SMOKE.exists(), reason="smoke data not prepared")
def test_resume_reproduces_uninterrupted_run(tmp_path):
    cpu = torch.device("cpu")
    a = train(tiny_cfg(final_name="a"), 0, cpu, ckpt_dir=tmp_path / "A", results_dir=tmp_path)
    b_cfg = tiny_cfg(final_name="b")
    stopped = train(b_cfg, 0, cpu, ckpt_dir=tmp_path / "B", results_dir=tmp_path, stop_at=6)
    assert stopped["stopped_at"] == 6
    b = train(b_cfg, 0, cpu, ckpt_dir=tmp_path / "B", results_dir=tmp_path, resume=True)
    sa, sb = load_checkpoint(a["final_checkpoint"])["model"], load_checkpoint(b["final_checkpoint"])["model"]
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


@pytest.mark.skipif(not SMOKE.exists(), reason="smoke data not prepared")
def test_protected_final_is_read_only_and_never_overwritten(tmp_path):
    cpu = torch.device("cpu")
    cfg = tiny_cfg(final_name="ref", protect_final=True, max_steps=2)
    s = train(cfg, 0, cpu, ckpt_dir=tmp_path, results_dir=tmp_path)
    p = Path(s["final_checkpoint"])
    assert not os.access(p, os.W_OK)
    assert (tmp_path / "ref_hashes.json").exists()
    with pytest.raises(FileExistsError):
        train(cfg, 0, cpu, ckpt_dir=tmp_path, results_dir=tmp_path)


@pytest.mark.skipif(not SMOKE.exists(), reason="smoke data not prepared")
def test_protect_survives_filesystem_without_chmod(tmp_path, monkeypatch):
    """Google Drive in Colab rejects chmod: the run must still finish and record the SHA."""
    import unlearnlang.train as tr

    def no_chmod(*a, **k):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(tr.os, "chmod", no_chmod)
    cfg = tiny_cfg(final_name="ref_long", protect_final=True, max_steps=2, hashes_file="ref_hashes.json")
    s = train(cfg, 0, torch.device("cpu"), ckpt_dir=tmp_path / "ck", results_dir=tmp_path / "res")
    assert len(s["sha256"]) == 64
    assert "ref_long_s0.pt" in (tmp_path / "res" / "ref_hashes.json").read_text(encoding="utf-8")
