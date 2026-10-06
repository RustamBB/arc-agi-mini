"""Integrate the arcgen branch into the Kaggle ARC-AGI-2 notebook (offline, self-contained).

python tools/build_kaggle_notebook.py --src original.ipynb --out notebooks/arc_agi2_with_arcgen.ipynb [--model arcgen_gpt.pt]

Changes to the source notebook (nothing else is touched):
  * a cell that unpacks the arcgen package (code is embedded, no internet / dataset needed);
  * a cell, placed BEFORE the blocking Qwen run, that starts the verified operation-bank search in a background process
    (low priority, incremental output file) so it works while the GPUs work;
  * the run_search_hybrid() cell: after v2/v3, if neither accepted, an accepted arcgen prediction may take the attempt_2
    slot through the same promote_verified_operation(); attempt_1 stays the Qwen winner (the existing assert still holds);
    a new variant 'transformer_search_v3_arcgen' is added and scored next to the old ones.
"""
import argparse, base64, io, json, re, tarfile, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in sorted((ROOT / "arcgen").rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    return base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode()


def cell(kind, src):
    base = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def text(c):
    return "".join(c["source"])


def replace_once(src, old, new, what):
    if src.count(old) != 1:
        raise SystemExit(f"cannot patch {what}: expected exactly one match, found {src.count(old)}")
    return src.replace(old, new)


MD = """
## ArcGen verified operation-bank branch (experimental, symbolic, offline)

A third verified source next to Layer-ops v2 and Transformer-search v3: a 108-operation grid DSL (geometry, colour, objects,
lines, relational rules such as marker-recolouring, template stamping, symmetry/tiling repair) searched by a target-guided
beam search. It runs **in a background process while Qwen is working**, so it is not limited by the 240 s post-processing window.

A prediction is accepted only if (1) a program reproduces every demonstration, (2) leave-one-demo-out search with the same
operation skeleton predicts each held-out demo exactly, (3) programs found with different seeds agree on every query, and
(4) every query output is a valid grid that differs from its input. It may only fill attempt_2 when v2 and v3 accepted nothing;
Qwen's attempt_1 is never changed. Search is bounded, not exhaustive. **Off by default** (`ARC_ARCGEN=1` to enable): measured 0 of 86 public-evaluation tasks accepted.
Optional: attach `arcgen_gpt.pt` (the small program-proposal model) to bias the search; it is not required.
"""

LAUNCH = '''
import os, sys, json, subprocess, time
from pathlib import Path

ARCGEN_ENABLED = os.getenv("ARC_ARCGEN", "0") == "1"   # OFF by default: 0/120 public-eval tasks accepted (see docs/KAGGLE_INTEGRATION.md)
ARCGEN_OUT = "/kaggle/working/arcgen_predictions.jsonl"
ARCGEN_PROC = None
if ARCGEN_ENABLED:
    _root = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
    _file = _root / ("arc-agi_test_challenges.json" if os.getenv("KAGGLE_IS_COMPETITION_RERUN")
                     else "arc-agi_evaluation_challenges.json")
    _model = next(iter(sorted(Path("/kaggle/input").rglob("arcgen_gpt.pt"))), None) if Path("/kaggle/input").exists() else None
    if _file.is_file():
        _cmd = [sys.executable, "-m", "arcgen.kaggle_branch", "--challenges", str(_file), "--out", ARCGEN_OUT,
                "--workers", os.getenv("ARC_ARCGEN_WORKERS", "2"), "--seconds", os.getenv("ARC_ARCGEN_SECONDS", "45"),
                "--deadline", str(global_end_time - 300)] + (["--model", str(_model)] if _model else [])
        ARCGEN_PROC = subprocess.Popen(_cmd, stdout=open("arcgen_branch.log", "w"), stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONPATH": os.getcwd(), "OMP_NUM_THREADS": "1"})
        print("arcgen branch started in background:", " ".join(_cmd[2:6]), "| model:", _model)
    else:
        print("arcgen branch disabled: challenge file not found", _file)
'''

WAIT = '''
from arcgen.kaggle_branch import load_results

ARCGEN_RESULTS = {}
if ARCGEN_ENABLED:
    _limit = time.time() + float(os.getenv("ARC_ARCGEN_WAIT_SECONDS", "600"))
    if "global_end_time" in globals():
        _limit = min(_limit, global_end_time + 400)
    while ARCGEN_PROC is not None and ARCGEN_PROC.poll() is None and time.time() < _limit:
        time.sleep(5)
    if ARCGEN_PROC is not None and ARCGEN_PROC.poll() is None:
        print("arcgen branch still running; using the partial results written so far")
        ARCGEN_PROC.terminate()
    ARCGEN_RESULTS = load_results(ARCGEN_OUT)
    print(f"arcgen branch: {len(ARCGEN_RESULTS)} tasks processed, "
          f"{sum(r['accepted'] for r in ARCGEN_RESULTS.values())} accepted")
'''


def patch_hybrid(src):
    src = replace_once(
        src,
        "        if original: assert current and _same_grid(current[0], original[0]), 'Qwen attempt_1 must stay unchanged'\n        combined[basekey] = current\n",
        "        no_arcgen = list(current)\n"
        "        arcgen_changed, arcgen_audit = False, {'accepted': False, 'reason': 'disabled_or_missing'}\n"
        "        _ag = globals().get('ARCGEN_RESULTS', {}).get(task_id)\n"
        "        if _ag is not None:\n"
        "            arcgen_audit = dict(_ag['audit'])\n"
        "            if (_ag['accepted'] and not prior_audit.get('accepted') and not search_audit.get('accepted')\n"
        "                    and query_index < len(_ag['predictions'])):\n"
        "                current, arcgen_changed = promote_verified_operation(original, current, _ag['predictions'][query_index])\n"
        "        if original: assert current and _same_grid(current[0], original[0]), 'Qwen attempt_1 must stay unchanged'\n"
        "        combined[basekey] = no_arcgen; with_arcgen[basekey] = current\n",
        "hybrid combine")
    src = replace_once(src, "    critic_only, v2_only, combined, audit, pools = {}, {}, {}, {}, {}\n",
                       "    critic_only, v2_only, combined, audit, pools = {}, {}, {}, {}, {}\n    with_arcgen = {}\n", "init dicts")
    src = replace_once(src, "'attempt_1_preserved': not original or _same_grid(current[0], original[0])}",
                       "'arcgen': arcgen_audit, 'arcgen_changed_attempt_2': arcgen_changed,\n"
                       "                         'attempt_1_preserved': not original or _same_grid(current[0], original[0])}", "audit entry")
    src = replace_once(src, "'full_public_eval_requested':",
                       "'arcgen_tasks': len(globals().get('ARCGEN_RESULTS', {})),\n"
                       "                'arcgen_accepted_tasks': sum(bool(r.get('accepted')) for r in globals().get('ARCGEN_RESULTS', {}).values()),\n"
                       "                'full_public_eval_requested':", "coverage")
    src = replace_once(src, "    return baseline, critic_only, v2_only, combined, audit, pools, coverage\n",
                       "    return baseline, critic_only, v2_only, combined, audit, pools, coverage, with_arcgen\n", "return")
    return src


def patch_final(src):
    src = replace_once(src, "baseline, critic, previous, hybrid, audit, pools, coverage = run_search_hybrid(decoder, data)",
                       "baseline, critic, previous, hybrid, audit, pools, coverage, hybrid_arcgen = run_search_hybrid(decoder, data)", "unpack")
    src = replace_once(src, "'transformer_search_v3': data.get_submission(hybrid)}",
                       "'transformer_search_v3': data.get_submission(hybrid),\n"
                       "               'transformer_search_v3_arcgen': data.get_submission(hybrid_arcgen)}", "submissions")
    src = replace_once(src, "    ('submission.json', submissions['transformer_search_v3']),",
                       "    ('submission.json', submissions['transformer_search_v3_arcgen' if ARCGEN_ENABLED else 'transformer_search_v3']),\n"
                       "    ('submission_v3_without_arcgen.json', submissions['transformer_search_v3']),", "files")
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nb = json.load(open(a.src))
    cells = nb["cells"]
    idx_run = next(i for i, c in enumerate(cells) if "python starter.py" in text(c) and c["cell_type"] == "code")
    idx_hybrid = next(i for i, c in enumerate(cells) if "def run_search_hybrid" in text(c))
    idx_final = next(i for i, c in enumerate(cells) if "= run_search_hybrid(decoder" in text(c))
    cells[idx_hybrid]["source"] = patch_hybrid(text(cells[idx_hybrid])).splitlines(keepends=True)
    cells[idx_final]["source"] = patch_final(text(cells[idx_final])).splitlines(keepends=True)
    unpack = ('import base64, io, tarfile, zlib\n'
              f'ARCGEN_B64 = "{bundle()}"\n'
              'tarfile.open(fileobj=io.BytesIO(zlib.decompress(base64.b64decode(ARCGEN_B64)))).extractall(".")\n'
              'import sys; sys.path.insert(0, ".")\n'
              'import arcgen; print("arcgen unpacked:", len(arcgen.OPS), "operations")\n')
    # order matters: wait-cell goes right before the final cell, launch cells right before the blocking Qwen run
    cells.insert(idx_final, cell("code", WAIT))
    cells[idx_run:idx_run] = [cell("markdown", MD), cell("code", unpack), cell("code", LAUNCH)]
    for c in cells:  # outputs of the source run would be stale for the new code
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["cells"] = cells
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(a.out, "w"), indent=1)
    print("wrote", a.out, "cells:", len(cells), "bundle KB:", len(bundle()) // 1024)


main()
