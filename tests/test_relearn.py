from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from unlearnlang.model import GPT, ModelConfig
from unlearnlang.relearn import relearn
from unlearnlang.utils import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "smoke"


@pytest.mark.skipif(not (SMOKE / "stats.json").exists(), reason="smoke data not prepared")
@pytest.mark.parametrize("cond", ["n0", "fresh1M"])
def test_relearn_logs_curve(tmp_path, cond):
    mcfg = dict(block_size=32, n_layer=1, n_head=2, d_model=16, dropout=0.0)
    torch.manual_seed(0)
    p = tmp_path / "m.pt"
    save_checkpoint(p, {"model": GPT(ModelConfig(vocab_size=62, **mcfg)).state_dict(), "config": {"model": mcfg}, "vocab_size": 62})
    log = relearn(str(p), "x", cond, 0, torch.device("cpu"), data_dir=SMOKE, results_dir=tmp_path, steps=4, eval_every=2,
                  block_size=32, eval_batches=2, eval_batch_size=2)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert [r["step"] for r in recs if "step" in r] == [0, 2, 4]
    assert "summary" in recs[-1]
