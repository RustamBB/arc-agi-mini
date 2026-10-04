"""Offline test of the integrated notebook: runs the REAL notebook cells (writefile modules, relation critic, v2 ops,
patched run_search_hybrid) with a fake Qwen decoder, on held-out demos of real training tasks (answers known).

python tools/test_kaggle_integration.py notebooks/arc_agi2_with_arcgen.ipynb challenges.json [--tasks 40] [--seconds 6]
Checks: attempt_1 never changes, arcgen only fills attempt_2 when v2/v3 accepted nothing, and reports how often the
arcgen attempt_2 is the true answer."""
import argparse, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("notebook"); ap.add_argument("challenges")
ap.add_argument("--tasks", type=int, default=40); ap.add_argument("--seconds", type=float, default=6)
a = ap.parse_args()
repo = Path(__file__).resolve().parent.parent
work = Path(tempfile.mkdtemp(prefix="kg_"))
os.chdir(work); sys.path.insert(0, str(work))
nb = json.load(open(repo / a.notebook if not Path(a.notebook).is_absolute() else a.notebook))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for src in cells:                                      # 1. materialise all %%writefile modules
    if src.startswith("%%writefile "):
        name, body = src.split("\n", 1)
        p = work / name.split()[1]; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(body)
(work / "operation_mining" / "__init__.py").touch()
ns = {"__name__": "nbtest", "global_end_time": time.time() + 3600}
def run_cell(match):
    src = next(s for s in cells if match in s and not s.startswith(("%%writefile", "!")))
    exec(compile(src, "<cell>", "exec"), ns)
run_cell("import base64, io, tarfile, zlib")           # arcgen bundle
for m in ("def find_search_checkpoint", "def calibrate_relation_critic", "def promote_verified_operation",
          "def run_search_hybrid"):
    run_cell(m)

# 2. fake queries: hold out the last demo of each real task (>=3 demos) as the "test"
data = json.load(open(a.challenges))
tasks = [(k, v) for k, v in data.items() if len(v["train"]) >= 3][: a.tasks]
class DS:  # minimal stand-ins for ArcDataset / ArcDecoder
    pass
raw, queries, truth = DS(), {}, {}
raw.queries = {}
for tid, t in tasks:
    demos, held = t["train"][:-1], t["train"][-1]
    raw.queries[tid] = {"train": demos, "test": [{"input": held["input"]}]}
    queries[f"{tid}_0"] = {"train": demos, "test": [{"input": held["input"]}]}
    truth[f"{tid}_0"] = np.array(held["output"])
decoder = DS(); decoder.dataset = DS(); decoder.dataset.queries = queries
rng = np.random.default_rng(0)
# fake Qwen candidates: the input copy first (a plausible wrong guess), a second wrong guess
decoder.run_selection_algo = lambda: {k: [np.array(q["test"][0]["input"]), np.rot90(np.array(q["test"][0]["input"]))]
                                      for k, q in queries.items()}
# 3. real branch results for the same fake queries (as the background process would produce)
from arcgen.kaggle_branch import verified_arcgen_predictions
t0 = time.time(); res = {}
for tid, t in tasks:
    preds, au = verified_arcgen_predictions(raw.queries[tid], time_limit=a.seconds)
    res[tid] = {"task_id": tid, "accepted": bool(au["accepted"]), "audit": au, "predictions": [np.asarray(p) for p in preds]}
ns["ARCGEN_RESULTS"] = res
print(f"branch precompute: {time.time()-t0:.0f}s, accepted {sum(r['accepted'] for r in res.values())}/{len(res)}")
os.environ["ARC_OPERATION_SECONDS_PER_TASK"] = "1"
out = ns["run_search_hybrid"](decoder, raw)
baseline, critic, v2, hyb_no, audit, pools, cov, hyb_ag = out
changed = ok = bad = 0
for k in queries:
    assert np.array_equal(hyb_ag[k][0], baseline[k][0]), "attempt_1 changed!"
    assert np.array_equal(hyb_no[k][0], baseline[k][0])
    if audit[k]["arcgen_changed_attempt_2"]:
        changed += 1
        ok += int(np.array_equal(hyb_ag[k][1], truth[k])); bad += int(not np.array_equal(hyb_ag[k][1], truth[k]))
        assert not audit[k]["v2"].get("accepted") and not audit[k]["search"].get("accepted")
print("coverage:", {k: v for k, v in cov.items() if "arcgen" in k or k in ("tasks", "queries")})
v2_ok = sum(bool(audit[k]["v2"].get("accepted")) for k in queries)
print(f"attempt_1 preserved for all {len(queries)} queries | v2 accepted: {v2_ok} | arcgen changed attempt_2: {changed} "
      f"(true answer: {ok}, wrong: {bad})")
