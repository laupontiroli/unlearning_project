"""Figures and the results table (PLAN §9.6).

    python scripts/plots.py curves --runs stage1_long_s0          training curves (fig. 4)
    python scripts/plots.py trajectory --seed 0                    unlearning trajectories (fig. 1)
    python scripts/plots.py relearn --seed 0                       relearning curves (fig. 2)
    python scripts/plots.py table --names ref_long_s0 theta_b_s0   results table (fig. 3) from results/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # macOS may hide uv's editable .pth file

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"pt_val_bpc": "#1f77b4", "pt_train_bpc": "#1f77b4", "en_val_bpc": "#d62728", "en_forget_train_bpc": "#d62728"}


def load_run(run: str) -> list[dict]:
    lines = Path(f"results/train_{run}.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if "step" in r]


def curves(runs: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, run in enumerate(runs):  # runs differ by marker
        recs = load_run(run)
        steps = [r["step"] for r in recs]
        for key in ("pt_train_bpc", "pt_val_bpc", "en_val_bpc", "en_forget_train_bpc"):
            if key in recs[-1]:
                ys = [r.get(key) for r in recs]
                ls = "--" if "train" in key else "-"
                ax.plot(steps, ys, ls, color=COLORS[key], marker="osv^D"[i % 5], ms=3, label=f"{run}: {key.replace('_bpc', '')}")
    ax.set_xlabel("step"); ax.set_ylabel("bits per character"); ax.set_ylim(bottom=1.0)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right", ncol=2)   # bottom area is empty (bpc > 1.5)
    ax.set_title("Training curves (fixed eval batches, dropout off)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


METHOD_COLORS = {"M1": "#2ca02c", "M2": "#d62728", "M3": "#1f77b4", "M3_oracle": "#9467bd", "theta_ref": "#000000",
                 "M3_proxy": "#ff7f0e", "M3_proxy_oracle": "#8c564b"}


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def trajectory(seed: int, out: Path, points: str | None = None) -> None:
    from unlearnlang.points import trajectory as traj_of
    pts = json.loads(Path(points or f"results/operating_points_s{seed}.json").read_text(encoding="utf-8"))
    ref = [r for r in read(Path(f"results/train_stage1_long_s{seed}.jsonl")) if "pt_val_bpc" in r][-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    styles = ["-", "--", ":", "-."]
    m3_i = 0
    for run, info in pts["runs"].items():
        t = traj_of(read(Path(f"results/unlearn_{run}.jsonl")))
        ls = "-"
        if info["method"] == "M3":
            ls, m3_i = styles[m3_i % 4], m3_i + 1
        lw = 2.2 if run in pts["selected"].values() else 1.0
        ax.plot([r["en_val_bpc"] for r in t], [r["pt_val_bpc"] for r in t], ls, lw=lw, marker=".", ms=4,
                color=METHOD_COLORS["M3_proxy" if "proxy" in run else info["method"]], label=run.rsplit("_s", 1)[0])
        p = info["practitioner"]
        ax.plot(p["en_val_bpc"], p["pt_val_bpc"], "o", mfc="none", mec=METHOD_COLORS["M3_proxy" if "proxy" in run else info["method"]], ms=9)
        if info["oracle"]:
            o = info["oracle"]
            ax.plot(o["en_val_bpc"], o["pt_val_bpc"], "*", color=METHOD_COLORS["M3_proxy" if "proxy" in run else info["method"]], ms=11)
    b = next(iter(pts["runs"].values()))["theta_b"]
    ax.plot(b["en_val_bpc"], b["pt_val_bpc"], "s", color="gray", ms=9, label="θ_b (start)")
    ax.plot(ref["en_val_bpc"], ref["pt_val_bpc"], "D", color="black", ms=8, label="θ_ref (target)")
    ax.axvline(ref["en_val_bpc"], color="black", lw=0.6, ls=":")
    import math
    ax.axvline(math.log2(62), color="gray", lw=0.8, ls="--")          # uniform guessing over the 62 symbols
    ax.text(math.log2(62), ax.get_ylim()[1], " uniform", fontsize=7, color="gray", va="top")
    ax.set_xlabel("English val bpc  (higher = more forgotten)")
    ax.set_ylabel("Portuguese val bpc  (lower = better kept)")
    ax.set_title("Unlearning trajectories (○ practitioner, ★ oracle point)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=160)
    print(f"saved {out}")


def relearn_plot(seed: int, out: Path, labels: list[str] | None = None) -> None:
    logs = sorted(Path("results").glob(f"relearn_*_s{seed}.jsonl"))
    if labels:
        logs = [p for p in logs if read(p)[0]["label"] in labels]
    else:                                  # main figure: the Gate 5 models only, no D8 proxy runs
        logs = [p for p in logs if "proxy" not in read(p)[0]["label"]]
    conds = sorted({read(p)[0]["condition"] for p in logs})
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 3.6), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        for p in logs:
            recs = [r for r in read(p) if "en_val_bpc" in r]
            if recs[0]["condition"] != cond:
                continue
            lab = recs[0]["label"]
            ax.plot([r["step"] for r in recs], [r["en_val_bpc"] for r in recs], "-" if lab == "theta_ref" else "--",
                    color=METHOD_COLORS.get(lab, "gray"), marker=".", label=lab)
        ax.set_title(f"relearning: {cond}", fontsize=10); ax.set_xlabel("fine-tuning step"); ax.set_ylabel("English val bpc")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=160)
    print(f"saved {out}")


def table(names: list[str], out: Path) -> None:
    allrecs = read(Path("results/eval.jsonl"))
    recs = {}
    for r in allrecs:                     # latest record per name, plus bpc of the same file from other records
        merged = {k: dict(v) if isinstance(v, dict) else v for k, v in r.items()}
        for o in allrecs:
            if o["ckpt_sha256"] == r["ckpt_sha256"] and o is not r:
                for key in ("bpc", "bpc_ci95"):
                    for split, v in o.get(key, {}).items():
                        merged.setdefault(key, {}).setdefault(split, v)
        recs[r["name"]] = merged
    cols = ["pt_test", "en_test", "pt_val", "en_val"]
    lines = ["| model | " + " | ".join(f"bpc {c}" for c in cols) + " | KL pt_test | KL en_test | EN-leak (PT prompts) | real-word (PT prompts) | EN-leak (EN prompts) |",
             "|" + "---|" * (len(cols) + 6)]
    for n in names:
        r = recs.get(n)
        if r is None:
            lines.append(f"| {n} | (not evaluated) |"); continue
        def cell(c):
            if c not in r["bpc"]:
                return "–"
            ci = r.get("bpc_ci95", {}).get(c)
            return f"{r['bpc'][c]:.3f}" + (f" ±{(ci[1] - ci[0]) / 2:.3f}" if ci else "")
        kl = r.get("kl_bits", {})
        g = r.get("generation", {})
        pp, ep = g.get("pt_prompts", {}), g.get("en_prompts", {})

        def num(d: dict, k: str, nd: int = 3) -> str:
            v = d.get(k)
            return "–" if v is None or v != v else f"{v:.{nd}f}"
        lines.append(f"| {n} | " + " | ".join(cell(c) for c in cols) +
                     f" | {num(kl, 'pt_test', 4)} | {num(kl, 'en_test', 4)}"
                     f" | {num(pp, 'en_leakage')} | {num(pp, 'real_word_rate')} | {num(ep, 'en_leakage')} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--out", default=None)
    t = sub.add_parser("trajectory"); t.add_argument("--seed", type=int, default=0); t.add_argument("--points", default=None); t.add_argument("--out", default=None)
    r = sub.add_parser("relearn"); r.add_argument("--seed", type=int, default=0); r.add_argument("--labels", nargs="+", default=None); r.add_argument("--out", default=None)
    tb = sub.add_parser("table"); tb.add_argument("--names", nargs="+", required=True); tb.add_argument("--out", default="results/table.md")
    args = ap.parse_args()
    if args.cmd == "curves":
        curves(args.runs, Path(args.out or f"results/figures/{'_'.join(args.runs)}_curves.png"))
    elif args.cmd == "trajectory":
        trajectory(args.seed, Path(args.out or f"results/figures/trajectory_s{args.seed}.png"), args.points)
    elif args.cmd == "relearn":
        relearn_plot(args.seed, Path(args.out or f"results/figures/relearn_s{args.seed}.png"), args.labels)
    elif args.cmd == "table":
        table(args.names, Path(args.out))


if __name__ == "__main__":
    main()
