"""How much headroom does answer *selection* have? Oracle accuracy of the Qwen candidate pool.

python tools/analyze_candidate_pool.py --pool qwen_candidate_pool.json --solutions arc-agi_evaluation_solutions.json
(qwen_candidate_pool.json is written by the Kaggle notebook: per query 'qwen_top8', 'hybrid_top2'.)
Prints task credit when the correct answer is accepted from the first k candidates (k=1,2,4,8): the gap between k=2
(what is submitted) and k=8 is what a better ranker could win; if k=8 ~ k=2, only better *generation* helps."""
import argparse, json
from collections import defaultdict
import numpy as np


def oracle(pool, sols, ks=(1, 2, 4, 8)):
    per = {k: defaultdict(list) for k in ks}
    for key, row in pool.items():
        tid, qi = key.rsplit("_", 1)
        sol = np.asarray(sols[tid][int(qi)])
        cands = row.get("qwen_top8", [])
        for k in ks:
            per[k][tid].append(any(np.array_equal(np.asarray(c), sol) for c in cands[:k]))
    out = {}
    for k in ks:  # credit = mean over a task's queries (queries missing from the pool count as wrong)
        out[k] = sum(sum(v) / len(sols[t]) for t, v in per[k].items())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True); ap.add_argument("--solutions", required=True)
    a = ap.parse_args()
    sols, pool = json.load(open(a.solutions)), json.load(open(a.pool))
    res = oracle(pool, sols)
    for k, v in res.items():
        print(f"correct answer within first {k} Qwen candidates: {v:6.2f} / {len(sols)}")
