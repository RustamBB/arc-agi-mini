"""Build notebooks/arcnet_kaggle.ipynb: train arcnet from scratch ON the task demos (test-time training) inside a time
budget, score on the public evaluation set, write candidates / a submission file. Self-contained (code embedded)."""
import base64, io, json, tarfile, zlib
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for pkg in ("arcgen", "arcnet"):
            for p in sorted((ROOT / pkg).rglob("*.py")):
                if "__pycache__" not in p.parts:
                    tar.add(p, arcname=str(p.relative_to(ROOT)))
    return base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode()


C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# arcnet on Kaggle - train from scratch on the demos, score on the public evaluation set

arcnet (colour layers -> objects -> cells, 14 shape relations, D4-invariant shape classes, class pooling, recursion,
LADDER curriculum) needs **no pretrained weights**: like TRM it learns each task's rule from the task's own demonstration
pairs (a per-task embedding) in the time you give it. This notebook trains for `MINUTES`, then:
1. prints the public-evaluation credit (2 attempts, voting over the trained augmentations) - the number to look at;
2. writes `arcnet_candidates.json` (top-8 grids with scores per test input) and `submission_arcnet.json`;
3. if a Qwen candidate pool (`qwen_candidate_pool.json`) is around, shows how many tasks arcnet solves that Qwen misses.

**Expectations (honest):** a from-scratch model trained for an hour will score low (single digits at best; literature
for comparable models after days on big GPUs is ~8 % on ARC-AGI-2). This run is a feasibility check: does it run on your GPU,
how fast, does the loss/exact-match climb, does LADDER's ladder get climbed. Only the *demonstration* pairs of evaluation
tasks are trained on; the evaluation solutions are used for scoring only.
""")
code(f'''
import base64, io, tarfile, zlib, sys, os
ARCNET_B64 = "{bundle()}"
tarfile.open(fileobj=io.BytesIO(zlib.decompress(base64.b64decode(ARCNET_B64)))).extractall(".")
sys.path.insert(0, ".")
import arcgen, arcnet
print("code unpacked")
''')
code('''
import json, math, pathlib, time
import numpy as np, torch

QUICK = bool(int(os.getenv("ARCNET_QUICK", "0")))        # tiny smoke-test configuration
MINUTES = float(os.getenv("ARCNET_MINUTES", "60"))       # total training wall-clock (70 % main, 30 % test-time phase)
if QUICK:
    cfg = dict(G=30, K=24, d=32, layers=1, heads=2, loops=2, aug=2, bs=2, limit=8)
    MINUTES = 0.7
else:                                                    # fits a 16 GB T4 / P100; raise d, layers, bs on bigger GPUs
    cfg = dict(G=30, K=64, d=256, layers=4, heads=8, loops=6, aug=16, bs=8, limit=0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = False
if DEVICE == "cuda":
    AMP = "bf16" if torch.cuda.is_bf16_supported() else "fp16"      # T4 / P100: no bf16
WORKERS = 0 if QUICK else max(1, min(4, (os.cpu_count() or 2) - 1))
ROOT = pathlib.Path(os.getenv("ARC_ROOT", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"))
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
f_train = ROOT / "arc-agi_training_challenges.json"
f_eval = ROOT / ("arc-agi_test_challenges.json" if RERUN else "arc-agi_evaluation_challenges.json")
f_sol = None if RERUN else ROOT / "arc-agi_evaluation_solutions.json"
print("device", DEVICE, "| amp", AMP, "| workers", WORKERS, "| minutes", MINUTES, "| files ok:", f_train.is_file(), f_eval.is_file())
''')
code('''
from arcnet.data import Sampler, evaluate, load_tasks, predict_all
from arcnet.ladder import LadderState, build_ladder, n_levels
from arcnet.model import LRRM
from arcnet.train import train_phase

train_tasks = load_tasks(str(f_train))
eval_tasks = load_tasks(str(f_eval), str(f_sol) if f_sol and f_sol.is_file() else None)
if cfg["limit"]:
    train_tasks, eval_tasks = train_tasks[:cfg["limit"]], eval_tasks[:cfg["limit"]]
tasks = train_tasks + eval_tasks
idx = {t["id"]: i for i, t in enumerate(tasks)}
eval_idx = [idx[t["id"]] for t in eval_tasks]
build_ladder(tasks, mode="compositional", kmax=4, comp_mode="both")
state = LadderState(len(tasks), n_levels("compositional", 4))
lv = {}
for t in tasks:
    for l, v in t["variants"].items(): lv[l] = lv.get(l, 0) + len(v)
print(f"{len(train_tasks)} training + {len(eval_tasks)} evaluation tasks | ladder variants per level {dict(sorted(lv.items()))}")
''')
code('''
model = LRRM(len(tasks), G=cfg["G"], K=cfg["K"], d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"],
             loops=cfg["loops"], A=cfg["aug"]).to(DEVICE)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
have_sol = any(t["solutions"] for t in eval_tasks)
def eval_fn():
    if not have_sol: return "no solutions (rerun mode)"
    credit, n = evaluate(model, eval_tasks, idx, DEVICE)
    return f"public-eval credit {credit:.2f}/{n}"
t0 = time.time()
quiet = (lambda m: print(m) if QUICK or "eval" in m or "probe" in m or int(m.split("step ")[1].split()[0]) % 200 == 0 else None)
opt = train_phase(model, tasks, Sampler(tasks, cfg["G"], cfg["K"], 0, state, A=cfg["aug"]), 10**9, cfg["bs"], 3e-4, DEVICE,
                  AMP, WORKERS, state, None, 200, eval_fn, 10**9 if QUICK else 3000, quiet, "main",
                  time_budget=MINUTES * 60 * 0.7)
print("--- test-time phase: evaluation tasks only (demos + their ladder variants)")
train_phase(model, tasks, Sampler(tasks, cfg["G"], cfg["K"], 1, state, only=eval_idx, A=cfg["aug"]), 10**9, cfg["bs"], 1.5e-4,
            DEVICE, AMP, WORKERS, state, eval_idx, 200, eval_fn, 10**9 if QUICK else 3000, quiet, "ttrl",
            time_budget=MINUTES * 60 * 0.3)
torch.save({"cfg": model.cfg, "state": model.state_dict(), "ids": [t["id"] for t in tasks]}, "arcnet.pt")
print(f"training done in {(time.time()-t0)/60:.1f} min")
''')
code('''
cands, sub = predict_all(model, eval_tasks, idx, DEVICE, top=8)
json.dump(cands, open("arcnet_candidates.json", "w")); json.dump(sub, open("submission_arcnet.json", "w"))
print("wrote arcnet_candidates.json, submission_arcnet.json")
if have_sol:
    eq = lambda a, b: np.array_equal(np.asarray(a), np.asarray(b))
    rows = []
    for t in eval_tasks:
        for qi, sol in enumerate(t["solutions"]):
            cs = [c["grid"] for c in cands[t["id"]][qi]]
            rows.append((t["id"], eq(cs[0], sol) if cs else False, any(eq(c, sol) for c in cs[:2]), any(eq(c, sol) for c in cs)))
    n = len(rows)
    print(f"queries: {n} | top-1 right {sum(r[1] for r in rows)} | top-2 right {sum(r[2] for r in rows)} | in top-8 {sum(r[3] for r in rows)}")
    here = [pathlib.Path("qwen_candidate_pool.json"), pathlib.Path("/kaggle/working/qwen_candidate_pool.json")]
    pool_f = next((p for p in here if p.is_file()), None)
    if pool_f is None and pathlib.Path("/kaggle/input").exists():
        pool_f = next(iter(pathlib.Path("/kaggle/input").rglob("qwen_candidate_pool.json")), None)
    if pool_f:                                           # complementarity with the Qwen pipeline
        pool = json.load(open(pool_f))
        sols = {t["id"]: t["solutions"] for t in eval_tasks}
        qwen_ok, arc_ok = set(), set()
        for key, row in pool.items():
            tid, qi = key.rsplit("_", 1)
            if tid in sols and any(eq(c, sols[tid][int(qi)]) for c in row["hybrid_top2"]):
                qwen_ok.add(key)
        arc_ok = {f"{t['id']}_{qi}" for t in eval_tasks for qi, sol in enumerate(t["solutions"])
                  if any(eq(c["grid"], sol) for c in cands[t["id"]][qi][:2])}
        print(f"queries right: Qwen hybrid {len(qwen_ok)} | arcnet {len(arc_ok)} | arcnet-only {len(arc_ok - qwen_ok)} "
              f"| both {len(arc_ok & qwen_ok)}  (arcnet-only queries are what an ensemble could add)")
    else:
        print("no qwen_candidate_pool.json found - skipping the complementarity check")
''')
nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, ROOT / "notebooks" / "arcnet_kaggle.ipynb")
print("wrote notebooks/arcnet_kaggle.ipynb")
