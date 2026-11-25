"""PLAN §11 rule 4: unlearning code must never touch the pre-Stage-2 reference model."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from unlearnlang.unlearn import check_init, unlearn, unlearn_loss

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"ref_s\d|ref_s\{|ref_long|checkpoints/ref|θ_ref|theta_ref|unlearnlang\.(evaluate|relearn)")
SMOKE = ROOT / "data" / "smoke" / "stats.json"


def module_closure(start: list[Path]) -> set[Path]:
    """The given files plus every unlearnlang module they import, transitively."""
    seen, todo = set(), list(start)
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        for m in re.findall(r"^\s*from unlearnlang\.(\w+) import|^\s*import unlearnlang\.(\w+)", p.read_text(encoding="utf-8"), re.M):
            name = m[0] or m[1]
            todo.append(ROOT / "src" / "unlearnlang" / f"{name}.py")
    return seen


def test_unlearning_code_never_names_the_reference():
    files = module_closure([ROOT / "src/unlearnlang/unlearn.py", ROOT / "scripts/unlearn.py"]) | {ROOT / "configs/unlearn.yaml", ROOT / "configs/unlearn_v2.yaml", ROOT / "configs/unlearn_proxy.yaml"}
    assert ROOT / "src/unlearnlang/train.py" in files      # the closure really follows imports
    for f in files:
        hits = FORBIDDEN.findall(f.read_text(encoding="utf-8"))
        assert not hits, f"{f.relative_to(ROOT)} mentions {hits}"


def test_refuses_reference_as_start():
    for bad in ("checkpoints/ref_long_s0.pt", "checkpoints/ref_s0.pt", "checkpoints/stage1_long_s0/last.pt"):
        with pytest.raises(ValueError):
            check_init(bad)
    check_init("checkpoints/theta_b_s0.pt")


def test_loss_signs():
    le, lp = torch.tensor(2.0), torch.tensor(3.0)
    assert unlearn_loss("M1", le, lp, 1.0) == 3.0
    assert unlearn_loss("M2", le, lp, 1.0) == -2.0
    assert unlearn_loss("M3", le, lp, 2.0) == 4.0


def _theta_b(tmp_path) -> Path:
    from unlearnlang.model import GPT, ModelConfig
    from unlearnlang.utils import save_checkpoint
    mcfg = dict(vocab_size=62, block_size=32, n_layer=1, n_head=2, d_model=16, dropout=0.0)
    torch.manual_seed(0)
    m = GPT(ModelConfig(**mcfg))
    p = tmp_path / "theta_b_s0.pt"
    save_checkpoint(p, {"model": m.state_dict(), "config": {"model": {k: v for k, v in mcfg.items() if k != "vocab_size"}},
                        "vocab_size": 62})
    return p


def _cfg(init: Path, **kw) -> dict:
    cfg = dict(data_dir=str(ROOT / "data" / "smoke"), init_from=str(init), block_size=32, en_batch=4, pt_batch=4,
               betas=[0.9, 0.95], grad_clip=1.0, max_steps=6, eval_interval=3, eval_batches=2, eval_batch_size=2,
               method="M3", lr=1e-3, lam=1.0)
    cfg.update(kw)
    return cfg


@pytest.mark.skipif(not SMOKE.exists(), reason="smoke data not prepared")
def test_run_writes_trajectory_and_checkpoints(tmp_path):
    s = unlearn(_cfg(_theta_b(tmp_path)), 0, torch.device("cpu"), ckpt_dir=tmp_path / "ck", results_dir=tmp_path)
    assert s["steps"] == 6 and s["diverged_at"] is None
    assert sorted(p.name for p in (tmp_path / "ck/unlearn/M3_lr0.001_lam1_s0").iterdir()) == ["step_0003.pt", "step_0006.pt"]
    again = unlearn(_cfg(_theta_b(tmp_path)), 0, torch.device("cpu"), ckpt_dir=tmp_path / "ck", results_dir=tmp_path)
    assert again["skipped"]


@pytest.mark.skipif(not SMOKE.exists(), reason="smoke data not prepared")
def test_gradient_ascent_triggers_divergence_guard(tmp_path):
    cfg = _cfg(_theta_b(tmp_path), method="M2", lr=0.05, max_steps=200, eval_interval=50)
    s = unlearn(cfg, 0, torch.device("cpu"), ckpt_dir=tmp_path / "ck", results_dir=tmp_path)
    assert s["diverged_at"] is not None and s["diverged_at"] < 200
    log = (tmp_path / "unlearn_M2_lr0.05_s0.jsonl").read_text(encoding="utf-8")
    assert '"event": "diverged"' in log


def test_dense_eval_schedule():
    from unlearnlang.unlearn import is_eval_step
    cfg = {"eval_interval": 100, "max_steps": 1000, "dense_until": 100, "dense_interval": 5}
    assert [s for s in range(1, 301) if is_eval_step(s, cfg)] == list(range(5, 101, 5)) + [200, 300]
    assert not is_eval_step(7, {"eval_interval": 100, "max_steps": 1000})


@pytest.mark.skipif(not SMOKE.exists(), reason="smoke data not prepared")
def test_tagged_run_has_dense_checkpoints(tmp_path):
    cfg = _cfg(_theta_b(tmp_path), max_steps=12, eval_interval=10, dense_until=6, dense_interval=2, tag="v2")
    s = unlearn(cfg, 0, torch.device("cpu"), ckpt_dir=tmp_path / "ck", results_dir=tmp_path)
    assert s["run"] == "M3_lr0.001_lam1_v2_s0"
    names = sorted(p.name for p in (tmp_path / "ck/unlearn/M3_lr0.001_lam1_v2_s0").iterdir())
    assert names == ["step_0002.pt", "step_0004.pt", "step_0006.pt", "step_0010.pt", "step_0012.pt"]


def test_proxy_forget_split_rules(tmp_path):
    """D8: proxy English must come from en_relearn_fresh beyond the part reserved for relearning."""
    from unlearnlang.unlearn import forget_data
    with pytest.raises(ValueError, match="reserved"):
        forget_data(tmp_path, {"forget_split": "en_relearn_fresh", "forget_offset": 0})
    with pytest.raises(ValueError, match="forget_split"):
        forget_data(tmp_path, {"forget_split": "en_test"})


def test_proxy_config_matches_selected_m3():
    import yaml
    v2 = yaml.safe_load((ROOT / "configs/unlearn_v2.yaml").read_text(encoding="utf-8"))
    px = yaml.safe_load((ROOT / "configs/unlearn_proxy.yaml").read_text(encoding="utf-8"))
    assert px["forget_split"] == "en_relearn_fresh" and px["forget_offset"] >= 1_000_000
    for k, v in v2.items():
        if k != "grid":
            assert px[k] == v, k
    assert {**px["grid"][0], "tag": "v2"} in v2["grid"]
