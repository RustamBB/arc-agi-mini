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

**Hardware:** one training process per GPU (DDP via `torch.distributed.run`; 4x L4 -> global batch 64, bf16). If NCCL
misbehaves set `ARCNET_NPROC=1` (single GPU) or `NCCL_P2P_DISABLE=1`.

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
import json, math, pathlib, subprocess, time
import numpy as np, torch

QUICK = bool(int(os.getenv("ARCNET_QUICK", "0")))        # tiny smoke-test configuration (CPU, 2 processes, ~1 min)
MINUTES = float(os.getenv("ARCNET_MINUTES", "90"))       # whole training budget (70 % main phase, 30 % test-time phase)
NGPU = torch.cuda.device_count()
NPROC = int(os.getenv("ARCNET_NPROC", NGPU if NGPU else (2 if QUICK else 1)))     # one training process per GPU
if QUICK:
    MINUTES = 0.8
    flags = "--G 30 --K 24 --dim 32 --layers 1 --heads 2 --loops 2 --aug 2 --bs 2 --limit-tasks 8 --workers 0 --probe-every 20 --eval-every 100000"
elif NGPU >= 4:                                          # 4 x L4 (24 GB, bf16): global batch 64
    flags = "--G 30 --K 64 --dim 384 --layers 4 --heads 8 --loops 6 --aug 16 --bs 16 --lr 4e-4 --probe-every 200 --eval-every 4000"
else:                                                    # 1-2 smaller GPUs (T4/P100: fp16)
    flags = "--G 30 --K 64 --dim 256 --layers 4 --heads 8 --loops 6 --aug 16 --bs 8 --lr 3e-4 --probe-every 200 --eval-every 3000"
amp = ""
if NGPU:
    amp = "--amp bf16" if torch.cuda.is_bf16_supported() else "--amp fp16"
cpus = os.cpu_count() or 2
workers = "" if QUICK else f"--workers {max(1, min(3, (cpus - 1) // max(1, NPROC)))}"
ROOT = pathlib.Path(os.getenv("ARC_ROOT", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"))
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
f_train = ROOT / "arc-agi_training_challenges.json"
f_eval = ROOT / ("arc-agi_test_challenges.json" if RERUN else "arc-agi_evaluation_challenges.json")
f_sol = None if RERUN else ROOT / "arc-agi_evaluation_solutions.json"
print(f"GPUs {NGPU} | processes {NPROC} | cpus {cpus} | amp '{amp}' | minutes {MINUTES} | files ok: {f_train.is_file()} {f_eval.is_file()}")
''')
code('''
cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={NPROC}", "--master_port=29631", "-m", "arcnet.train",
       "--train-challenges", str(f_train), "--eval-challenges", str(f_eval), "--out", "arcnet.pt", "--outdir", ".",
       "--ladder", "--ladder-mode", "compositional", "--ladder-order", "both", "--ttrl-steps", "10000000000",
       "--steps", "10000000000", "--minutes", str(MINUTES)] + flags.split() + amp.split() + workers.split()
if f_sol and f_sol.is_file():
    cmd += ["--eval-solutions", str(f_sol)]
env = {**os.environ, "PYTHONPATH": os.getcwd(), "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
keep = ("ladder:", "parameters", "eval", "probe", "TTRL", "final", "wrote", "Error", "error", "Traceback", "NCCL")
for line in proc.stdout:                                  # print only the useful lines (+ every 200th training step)
    if line.startswith(keep) or any(k in line for k in keep) or (QUICK and "step" in line) or \
            ("] step " in line and int(line.split("step ")[1].split()[0]) % 200 == 0):
        print(line.rstrip())
rc = proc.wait()
print(f"training process exited with code {rc} after {(time.time()-t0)/60:.1f} min")
assert rc == 0, "training failed - see the lines above (if NCCL hangs/errors: set ARCNET_NPROC=1 or NCCL_P2P_DISABLE=1)"
''')
code('''
from arcnet.data import load_tasks
eval_tasks = load_tasks(str(f_eval), str(f_sol) if f_sol and f_sol.is_file() else None)
if QUICK:
    eval_tasks = eval_tasks[:8]
cands = json.load(open("arcnet_candidates.json"))
have_sol = any(t["solutions"] for t in eval_tasks)
print("candidates for", len(cands), "tasks written to arcnet_candidates.json / submission_arcnet.json / arcnet.pt")
if have_sol:
    eq = lambda a, b: np.array_equal(np.asarray(a), np.asarray(b))
    rows = []
    for t in eval_tasks:
        for qi, sol in enumerate(t["solutions"]):
            cs = [c["grid"] for c in cands[t["id"]][qi]]
            rows.append((t["id"], qi, eq(cs[0], sol) if cs else False, any(eq(c, sol) for c in cs[:2]), any(eq(c, sol) for c in cs)))
    print(f"queries: {len(rows)} | top-1 right {sum(r[2] for r in rows)} | top-2 right {sum(r[3] for r in rows)} | in top-8 {sum(r[4] for r in rows)}")
    credit = {}
    for tid, qi, _, ok2, _ in rows:
        credit.setdefault(tid, []).append(ok2)
    print(f"task credit (2 attempts): {sum(np.mean(v) for v in credit.values()):.2f} / {len(credit)}")
    here = [pathlib.Path("qwen_candidate_pool.json"), pathlib.Path("/kaggle/working/qwen_candidate_pool.json")]
    pool_f = next((p for p in here if p.is_file()), None)
    if pool_f is None and pathlib.Path("/kaggle/input").exists():
        pool_f = next(iter(pathlib.Path("/kaggle/input").rglob("qwen_candidate_pool.json")), None)
    if pool_f:                                           # complementarity with the Qwen pipeline
        pool = json.load(open(pool_f))
        sols = {t["id"]: t["solutions"] for t in eval_tasks}
        qwen_ok = {k for k, row in pool.items() if k.rsplit("_", 1)[0] in sols
                   and any(eq(c, sols[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])]) for c in row["hybrid_top2"])}
        arc_ok = {f"{tid}_{qi}" for tid, qi, _, ok2, _ in rows if ok2}
        print(f"queries right: Qwen hybrid {len(qwen_ok)} | arcnet {len(arc_ok)} | arcnet-only {len(arc_ok - qwen_ok)} "
              f"| both {len(arc_ok & qwen_ok)}  (arcnet-only queries are what an ensemble could add)")
    else:
        print("no qwen_candidate_pool.json found - skipping the complementarity check")
''')
nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, ROOT / "notebooks" / "arcnet_kaggle.ipynb")
print("wrote notebooks/arcnet_kaggle.ipynb")
