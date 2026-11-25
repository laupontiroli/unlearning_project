from __future__ import annotations

import json

from unlearnlang.points import operating_points, select_points, trajectory


def rec(step, pt, en):
    return {"step": step, "ckpt": f"c{step}", "pt_val_bpc": pt, "en_val_bpc": en}


def test_practitioner_stops_before_2pct_and_oracle_first_reach():
    traj = [rec(0, 1.00, 2.0), rec(100, 1.01, 2.5), rec(200, 1.019, 2.9), rec(300, 1.03, 3.2), rec(400, 1.0, 3.5)]
    pts = operating_points(traj, ref_en=3.0)
    assert pts["practitioner"]["step"] == 200     # 300 breaks the 2% rule; later recovery does not count
    assert pts["oracle"]["step"] == 300
    assert operating_points(traj, ref_en=9.0)["oracle"] is None


def test_divergence_excludes_bad_models():
    log = [rec(0, 1.0, 2.0), rec(100, 1.0, 2.5), {"event": "diverged", "step": 100}, rec(99, 1.0, 2.4)]
    assert [r["step"] for r in trajectory(log)] == [0, 99]


def test_select_picks_m3_with_most_forgetting(tmp_path):
    (tmp_path / "train_stage1_long_s0.jsonl").write_text(json.dumps({"step": 6000, "en_val_bpc": 2.6}) + "\n")
    for run, en in (("M3_a_s0", 2.7), ("M3_b_s0", 3.1), ("M1_x_s0", 2.2)):
        lines = [rec(0, 1.0, 2.0), rec(100, 1.0, en), {"summary": {"run": run, "method": run[:2], "diverged_at": None}}]
        (tmp_path / f"unlearn_{run}.jsonl").write_text("\n".join(json.dumps(l) for l in lines))
    out = select_points(tmp_path, 0)
    assert out["selected"] == {"M1": "M1_x_s0", "M3": "M3_b_s0"}
