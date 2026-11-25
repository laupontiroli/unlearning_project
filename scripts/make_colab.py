"""Build the small Colab code bundle and the phase notebooks (docs/colab_pipeline.md).

    uv run python scripts/make_colab.py

Writes colab/code_bundle.tar.gz (code, configs, tests and a snapshot of results/ as results_mac/) and
colab/02_gate2_eval.ipynb, 03_stage2, 04_unlearn and 05_eval_relearn. The 60 MB data bundle (colab/unlearn_bundle.tar.gz)
is already on Drive and is not rebuilt: the data does not change. Re-run this script and re-upload
code_bundle.tar.gz whenever the code changes.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "colab"
CODE = ["pyproject.toml", "uv.lock", "src", "scripts", "configs", "tests"]


def skip(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
    # deviations.md is edited on the Mac only, so it is not copied to Drive.
    return None if "__pycache__" in ti.name or ti.name.endswith((".pyc", ".DS_Store", "deviations.md")) else ti


def build_code_bundle() -> Path:
    out = COLAB / "code_bundle.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for p in CODE:
            tar.add(ROOT / p, arcname=p, filter=skip)
        tar.add(ROOT / "results", arcname="results_mac", filter=skip)
    return out


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}}}


SETUP = [
    code("""
# 1. Settings + GPU check
SEED = 0
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code("""
# 2. Mount Drive. MyDrive/unlearn/{checkpoints,results} is the working copy from Phase 3 on.
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/unlearn'
import os
for d in ('checkpoints', 'results', 'gate_packs'):
    os.makedirs(f'{DRIVE}/{d}', exist_ok=True)
print(sorted(os.listdir(DRIVE)))
"""),
    code("""
# 3. Data bundle (unchanged since Phase 0) + latest code bundle on top; check the data SHAs;
#    link checkpoints/ and results/ to Drive so every script writes there directly.
!rm -rf /content/unlearn && mkdir -p /content/unlearn
!tar xzf {DRIVE}/unlearn_bundle.tar.gz -C /content/unlearn
!tar xzf {DRIVE}/code_bundle.tar.gz -C /content/unlearn
%cd /content/unlearn
!cd data/processed && sha256sum -c --quiet MANIFEST.sha256 && echo "data OK"
!rm -rf checkpoints results && ln -s {DRIVE}/checkpoints checkpoints && ln -s {DRIVE}/results results
"""),
    code("""
# 4. Dependencies + tests. tests/test_data.py is not run here: it needs the *_titles.txt files, which are
#    not in the bundle. It passed on the Mac at Gate 0, and cell 3 already proved the .bin files are
#    byte-identical to the Mac copy (SHA-256). Smoke-data tests are skipped (no smoke data on Colab).
!pip -q install pyyaml
!python -m pytest -q --ignore=tests/test_data.py 2>&1 | tail -2
"""),
    code("""
# 5. Bring the Mac's results into Drive (never overwrites), then re-hash every protected checkpoint.
#    Drive cannot chmod, so this SHA check is what protects θ_ref. It must print OK, never raise.
!python scripts/colab_sync.py merge --src results_mac --dst results
!python scripts/colab_sync.py verify
"""),
]


def gate_pack(n: int) -> dict:
    return code(f"""
# Gate pack: re-check the hashes, then zip results/ (small files only, no checkpoints).
# Download MyDrive/unlearn/gate_packs/gate{n}_results.zip and unzip it into the project as colab_results/gate{n}/
!python scripts/colab_sync.py verify
!cd {{DRIVE}} && rm -f gate_packs/gate{n}_results.zip && zip -qr gate_packs/gate{n}_results.zip results && ls -la gate_packs
""")


GATE2 = notebook([
    md("""
# Gate 2 (redo): evaluate θ_ref = ref_long on Colab

Run this after `stage1_long.ipynb` has finished (same or new session).
**Upload first:** `colab/code_bundle.tar.gz` to `MyDrive/unlearn/` (the data bundle is already there).
Evaluation is the same as for the pilot: 500 fixed chunks per split, 200 generations per language from
*validation* prompts. `pt_test` is used exactly once (logged in `results/test_access.jsonl`; a second
access is refused).
"""),
    *SETUP,
    code("""
# 6. Training curves (train vs val bpc every 500 steps)
!python scripts/plots.py curves --runs stage1_long_s0
from IPython.display import Image; Image('results/figures/stage1_long_s0_curves.png')
"""),
    code("""
# 7. Evaluate θ_ref (skipped if already done: pt_test may be read only once)
import os
if os.path.exists(f'results/samples/ref_long_s{SEED}.json'):
    print('already evaluated')
else:
    !python scripts/eval_checkpoint.py --ckpt checkpoints/ref_long_s{SEED}.pt --name ref_long_s{SEED} \\
        --splits pt_val en_val pt_train en_forget_train --test-splits pt_test --prompts val
!tail -1 results/eval.jsonl
"""),
    gate_pack(2),
])

STAGE2 = notebook([
    md("""
# Phase 3: Stage 2, θ_b (mixed 50/50) and θ_en (pure English)

**Only after Gate 2 is approved.** Upload the latest `colab/code_bundle.tar.gz` to `MyDrive/unlearn/`.
Both runs start from θ_ref = `checkpoints/ref_long_s{SEED}.pt` with a fresh optimizer.
- θ_b: 32 PT + 32 EN sequences per step, 1800 steps (`configs/stage2_mixed.yaml`).
- θ_en: 64 EN sequences per step, 500 steps (`configs/stage2_en.yaml`).

Every cell can be re-run after a disconnect: training resumes from its last checkpoint and finished
steps are skipped.
"""),
    *SETUP,
    code("""
# 6. θ_b: mixed Stage 2
![ -f checkpoints/theta_b_s{SEED}.pt ] && echo "theta_b done" || \\
  python scripts/train.py --config configs/stage2_mixed.yaml --seed {SEED} --resume 2>&1 | grep -v Warning
"""),
    code("""
# 7. θ_en: pure-English diagnostic
![ -f checkpoints/theta_en_s{SEED}.pt ] && echo "theta_en done" || \\
  python scripts/train.py --config configs/stage2_en.yaml --seed {SEED} --resume 2>&1 | grep -v Warning
"""),
    code("""
# 8. Evaluate both on validation + train splits (no test splits before Phase 5), validation prompts
import os
for m in ('theta_b', 'theta_en'):
    if os.path.exists(f'results/samples/{m}_s{SEED}.json'):
        print(m, 'already evaluated'); continue
    !python scripts/eval_checkpoint.py --ckpt checkpoints/{m}_s{SEED}.pt --name {m}_s{SEED} \\
        --splits pt_val en_val pt_train en_forget_train --prompts val
"""),
    code("""
# 9. Curves
!python scripts/plots.py curves --runs stage2_mixed_s{SEED} stage2_en_s{SEED}
from IPython.display import Image; Image(f'results/figures/stage2_mixed_s{SEED}_stage2_en_s{SEED}_curves.png')
"""),
    gate_pack(3),
])

UNLEARN = notebook([
    md("""
# Phase 4: unlearning English from θ_b (Gate 4)

**Only after Gate 3 is approved.** Upload the latest `colab/code_bundle.tar.gz` first.
Runs the 6-run grid of `configs/unlearn_v2.yaml` (deviations D5, D6): M1 control, M2 gradient ascent and
4 × M3 NegGrad+ (lr × λ). Each run takes at most 1000 steps. Evaluation and checkpoint every 5 steps for the first
100 steps, then every 100 steps. M1 is reused from the first grid.
The unlearning code never loads θ_ref (`tests/test_isolation.py`). The operating points are chosen afterwards
by `select_points.py`, which is evaluation-side code.
"""),
    *SETUP,
    code("""
# 6. Unlearning grid v2 (deviation D6: eval + checkpoint every 5 steps for the first 100 steps).
#    M1 is reused from v1 (skipped); M2 and M3 rerun with tag v2. Finished runs are skipped.
!python scripts/unlearn.py --grid --config configs/unlearn_v2.yaml --seed {SEED} 2>&1 | grep -v Warning
"""),
    code("""
# 7. Operating points + trajectory figure. Paste this cell's printed output into the chat for Gate 4.
!python scripts/select_points.py --seed {SEED} --config configs/unlearn_v2.yaml
!python scripts/plots.py trajectory --seed {SEED}
from IPython.display import Image; Image(f'results/figures/trajectory_s{SEED}.png')
"""),
    code("""
# 8. Samples at the practitioner point of every run (validation prompts, no test data yet)
import json, os
pts = json.load(open(f'results/operating_points_s{SEED}.json'))
done_ckpts = {f'checkpoints/theta_b_s{SEED}.pt': f'theta_b_s{SEED}'}   # θ_b was evaluated at Gate 3
for run, info in pts['runs'].items():
    ck = info['practitioner']['ckpt']
    if ck in done_ckpts:
        print(f'==== {run}: practitioner point = {done_ckpts[ck]} (step {info["practitioner"]["step"]}), not evaluated again')
        continue
    name = f'{run}@prac_val'
    done_ckpts[ck] = name
    if not os.path.exists(f'results/samples/{name}.json'):
        ck = info['practitioner']['ckpt']
        !python scripts/eval_checkpoint.py --ckpt {ck} --name {name} --splits pt_val en_val --prompts val 2>&1 | grep -v Warning
    s = json.load(open(f'results/samples/{name}.json'))
    print(f'==== {run} (step {info["practitioner"]["step"]})')
    for lang in ('pt', 'en'):
        print(f'  [{lang}] ...{s[lang][0]["prompt"][-30:]!r} -> {s[lang][0]["generation"][:160]!r}')
"""),
    gate_pack(4),
])

PHASE5 = notebook([
    md("""
# Phase 5: test evaluation and relearning attacks (Gate 5)

**Only after Gate 4 is approved.** Upload the latest `colab/code_bundle.tar.gz` first.
The test splits are read here for the first time, once per model (a second access is refused).
Relearning: θ_ref and the selected M1/M2/M3 models are fine-tuned for 300 steps, with Portuguese only (n0)
and with 1M characters of fresh English (fresh1M).
"""),
    *SETUP,
    code("""
# 6. Test-split evaluation of every model: bpc + 95% CI, KL to θ_ref, leakage from test prompts
!python scripts/phase5_eval.py --seed {SEED} 2>&1 | grep -v Warning
"""),
    code("""
# 7. Relearning attacks (finished runs are skipped)
!python scripts/relearn.py --seed {SEED} --with-oracle 2>&1 | grep -v Warning
"""),
    code("""
# 8. Results table + relearning figure. Paste the printed table into the chat for Gate 5.
import json
pts = json.load(open(f'results/operating_points_s{SEED}.json'))
names = [f'theta_ref_s{SEED}@test', f'theta_b_s{SEED}@test', f'theta_en_s{SEED}@test']
for run, info in pts['runs'].items():
    names.append(f'{run}@prac')
    if run in pts['selected'].values() and info['oracle']:
        names.append(f'{run}@oracle')
NAMES = ' '.join(names)
!python scripts/plots.py table --names {NAMES}
!python scripts/plots.py relearn --seed {SEED}
from IPython.display import Image; Image(f'results/figures/relearn_s{SEED}.png')
"""),
    gate_pack(5),
])


PROXY = notebook([
    md("""
# D8: unlearning with proxy English the model never saw

**Upload first:** the latest `colab/code_bundle.tar.gz` to `MyDrive/unlearn/`.
This runs the selected M3 configuration again (lr 5e-5, λ 2), this time using English that no model has trained on:
`en_relearn_fresh` from character 1M on. Then it evaluates the practitioner and oracle points on the test splits
(each read once) and runs both relearning attacks from them. About 30 min on a T4.
Every step stops with an error if its script fails.
"""),
    *SETUP,
    code("""
# 6. Helper: run a script, stream its output, and raise if it fails.
import subprocess, sys, json
def run(*args):
    p = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if 'Warning' not in line:
            print(line, end='')
    if p.wait():
        raise RuntimeError(f'failed with exit code {p.returncode}: {args}')
PROXY_POINTS = f'results/operating_points_proxy_s{SEED}.json'
"""),
    code("""
# 7. Proxy unlearning run (collapses within ~70 steps; dense eval every 5 steps)
run('scripts/unlearn.py', '--grid', '--config', 'configs/unlearn_proxy.yaml', '--seed', str(SEED))
"""),
    code("""
# 8. Operating points of the proxy run (own file: the Gate 4 selection is untouched), compared with the original M3
run('scripts/select_points.py', '--seed', str(SEED), '--config', 'configs/unlearn_proxy.yaml', '--out', PROXY_POINTS)
orig = json.load(open(f'results/operating_points_s{SEED}.json'))
prox = json.load(open(PROXY_POINTS))
both = {**prox, 'runs': {orig['selected']['M3']: orig['runs'][orig['selected']['M3']], **prox['runs']},
        'selected': {'M3': orig['selected']['M3'], 'M3p': prox['selected']['M3']}}
json.dump(both, open(f'results/operating_points_m3_vs_proxy_s{SEED}.json', 'w'), indent=1)
run('scripts/plots.py', 'trajectory', '--seed', str(SEED), '--points', f'results/operating_points_m3_vs_proxy_s{SEED}.json',
    '--out', f'results/figures/trajectory_proxy_s{SEED}.png')
from IPython.display import Image, display; display(Image(f'results/figures/trajectory_proxy_s{SEED}.png'))
"""),
    code("""
# 9. Test-split evaluation of the proxy run's practitioner and oracle points (earlier models are skipped)
run('scripts/phase5_eval.py', '--seed', str(SEED), '--points', PROXY_POINTS)
"""),
    code("""
# 10. Relearning attacks from the proxy run (theta_ref's runs are reused)
run('scripts/relearn.py', '--seed', str(SEED), '--points', PROXY_POINTS, '--suffix', 'proxy', '--with-oracle')
"""),
    code("""
# 11. Comparison table + relearning figure. Paste the table into the chat.
m3, m3p = orig['selected']['M3'], prox['selected']['M3']
names = [f'theta_ref_s{SEED}@test', f'theta_b_s{SEED}@test', f'{m3}@prac', f'{m3}@oracle', f'{m3p}@prac', f'{m3p}@oracle']
run('scripts/plots.py', 'table', '--names', *names, '--out', 'results/table_proxy.md')
run('scripts/plots.py', 'relearn', '--seed', str(SEED), '--labels', 'theta_ref', 'M3', 'M3_oracle', 'M3_proxy', 'M3_proxy_oracle',
    '--out', f'results/figures/relearn_proxy_s{SEED}.png')
display(Image(f'results/figures/relearn_proxy_s{SEED}.png'))
"""),
    gate_pack(6),
])


def main() -> None:
    print("wrote", build_code_bundle())
    for name, nb in (("02_gate2_eval", GATE2), ("03_stage2", STAGE2), ("04_unlearn", UNLEARN), ("05_eval_relearn", PHASE5), ("06_proxy_unlearn", PROXY)):
        p = COLAB / f"{name}.ipynb"
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", p)


if __name__ == "__main__":
    main()
