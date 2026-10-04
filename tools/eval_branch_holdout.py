"""Precision of the fail-closed Kaggle branch without solution files.
Each task with >= 3 demos: use all demos but the last, treat the last demo as the hidden test.
Reports how many tasks the branch accepts and how many accepted predictions are correct.
python tools/eval_branch_holdout.py challenges.json --seconds 10 --limit 300 [--workers 4] [--out r.json]"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arcgen.kaggle_branch import verified_arcgen_predictions  # noqa: E402


def work(item):
    tid, task, seconds = item
    demos, held = task["train"][:-1], task["train"][-1]
    preds, audit = verified_arcgen_predictions({"train": demos, "test": [{"input": held["input"]}]}, time_limit=seconds)
    correct = bool(preds) and np.array_equal(np.asarray(preds[0]), np.asarray(held["output"]))
    return tid, {"accepted": bool(audit["accepted"]), "correct": correct, "reason": audit.get("reason"),
                 "program": audit.get("program")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--limit", type=int, default=300); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = json.load(open(a.path))
    items = [(k, v, a.seconds) for k, v in d.items() if len(v["train"]) >= 3][: a.limit]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = dict(p.imap_unordered(work, items))
    acc = [k for k, v in res.items() if v["accepted"]]
    ok = [k for k in acc if res[k]["correct"]]
    print(f"{len(res)} tasks (>=3 demos), {time.time()-t0:.0f}s")
    print(f"accepted {len(acc)} ({100*len(acc)/len(res):.1f}%), correct among accepted {len(ok)} "
          f"-> precision {100*len(ok)/max(1,len(acc)):.1f}%; correct overall {100*len(ok)/len(res):.1f}%")
    for k in acc:
        if not res[k]["correct"]:
            print("  wrong:", k, res[k]["program"])
    if a.out:
        json.dump(res, open(a.out, "w"))
