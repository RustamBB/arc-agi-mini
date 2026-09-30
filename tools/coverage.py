"""How many real ARC tasks does the op bank solve with ONE op (params found by random search)?

usage: python tools/coverage.py challenges.json [--tries 40] [--limit N] [--out cov.json]
A task counts as covered if some op with sampled params maps every train input to its output.
Lower bound only (random parameter search, single op), but comparable between bank versions.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arcgen.core import OpError  # noqa: E402
from arcgen.ops import OPS  # noqa: E402
from arcgen.program import _plain, apply_step  # noqa: E402


def solve(item):
    tid, task, tries = item
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    rng = np.random.default_rng(0)
    a0, b0 = pairs[0]
    if max(a0.shape) > 30:
        return tid, None
    for name, o in OPS.items():
        for _ in range(tries):
            try:
                params = _plain(o.sample(rng, a0))
                if not (apply_step(a0, name, params) == b0).all():
                    continue
                if all((apply_step(a, name, params) == b).all() for a, b in pairs[1:]):
                    return tid, name
            except (OpError, ValueError, IndexError, TypeError, KeyError, AssertionError):
                continue
            except Exception:
                continue
    return tid, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = json.load(open(a.path))
    items = [(k, v, a.tries) for k, v in d.items()][: a.limit or None]
    t = time.time()
    with mp.Pool() as p:
        res = dict(p.map(solve, items, chunksize=4))
    hit = {k: v for k, v in res.items() if v}
    print(f"covered {len(hit)}/{len(res)} tasks ({100*len(hit)/len(res):.1f}%) in {time.time()-t:.0f}s")
    from collections import Counter
    print(Counter(hit.values()).most_common(25))
    if a.out:
        json.dump(res, open(a.out, "w"))
