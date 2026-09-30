"""Beam-search coverage of a real ARC set.
usage: python tools/search_cov.py challenges.json --out res.json [--depth 3] [--budget 30] [--limit N] [--offset K]
Writes {task_id: {"program": [...] | null, "err": residual cells summed over train pairs, "cells": total target cells, "best": [...]}}"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arcgen.search import search  # noqa: E402
from arcgen.program import to_text  # noqa: E402
from arcgen.serialize import check_program  # noqa: E402


def work(item):
    tid, task, kw = item
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    if max(max(a.shape + b.shape) for a, b in pairs) > 30:
        return tid, {"program": None, "err": None}
    prog, err, best = search(pairs, **kw)
    ok = bool(prog) and check_program(to_text(prog), pairs)
    return tid, {"program": prog if ok else None, "err": int(err),
                 "cells": int(sum(b.size for _, b in pairs)), "best": best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--budget", type=float, default=30)
    ap.add_argument("--tries", type=int, default=60); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = json.load(open(a.path))
    kw = dict(depth=a.depth, budget=a.budget, tries=a.tries, seed=a.seed)
    items = [(k, v, kw) for k, v in list(d.items())[a.offset:]][: a.limit or None]
    t = time.time()
    with mp.Pool() as p:
        res = dict(p.imap_unordered(work, items, chunksize=1))
    solved = {k for k, v in res.items() if v["program"]}
    from collections import Counter
    print(f"solved {len(solved)}/{len(res)} ({100*len(solved)/len(res):.1f}%) in {time.time()-t:.0f}s")
    print(Counter(s["op"] for k in solved for s in res[k]["program"]).most_common(30))
    json.dump(res, open(a.out, "w"))
