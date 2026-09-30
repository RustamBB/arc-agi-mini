"""Dataset generation and writing (ARC-AGI JSON + program annotations)."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np

from .ops import OPS
from .program import Task, build_task, sample_program, to_text

LEN_P = {1: .2, 2: .4, 3: .3, 4: .1}


def grid_str(g) -> str:
    return "\n".join("".join(str(int(v)) for v in row) for row in g)


def make_task(seed: int, index: int, op_pool: list[str], min_len=1, max_len=4) -> Task:
    """Deterministically build task number ``index`` (retries internally until valid)."""
    rng = np.random.default_rng([seed, index])
    lens = [n for n in LEN_P if min_len <= n <= max_len]
    p = np.array([LEN_P[n] for n in lens], dtype=float)
    p /= p.sum()
    for _ in range(200):
        length = int(rng.choice(lens, p=p))
        pal = [int(c) for c in rng.choice(np.arange(1, 10), int(rng.integers(2, 5)), replace=False)]
        lo = int(rng.integers(5, 11))
        hi = int(rng.integers(lo, 21))
        prog, gen = sample_program(rng, length, op_pool, pal, lo, hi)
        if prog is None:
            continue
        task = build_task(rng, prog, gen, pal, lo, hi,
                          n_train=int(rng.integers(2, 6)), n_test=int(rng.integers(1, 3)))
        if task is not None:
            return task
    raise RuntimeError("could not build a task")


def task_id(task: Task) -> str:
    h = hashlib.sha1()
    for a, b in task.train + task.test:
        h.update(a.tobytes() + bytes(a.shape) + b.tobytes() + bytes(b.shape))
    return h.hexdigest()[:8]


def arc_json(task: Task) -> dict:
    enc = lambda ps: [{"input": a.tolist(), "output": b.tolist()} for a, b in ps]
    return {"train": enc(task.train), "test": enc(task.test)}


def record(task: Task, tid: str, with_trace: bool) -> dict:
    rec = {
        "id": tid,
        **arc_json(task),
        "program": task.program,
        "program_text": to_text(task.program),
        "ops": [s["op"] for s in task.program],
        "categories": sorted({OPS[s["op"]].category for s in task.program}),
        "input_generator": task.gen,
    }
    if with_trace:
        rec["traces"] = [[g.tolist() for g in t] for t in task.traces]
    return rec


def resolve_ops(include=None, exclude=None, categories=None) -> list[str]:
    pool = [n for n, o in OPS.items()
            if (not include or n in include) and n not in (exclude or ())
            and (not categories or o.category in categories)]
    if not pool:
        raise ValueError("no operations selected")
    return sorted(pool)


def _work(args):
    seed, i, pool, lo, hi = args
    try:
        return make_task(seed, i, pool, lo, hi)
    except RuntimeError:
        return None


def _tasks(seed, op_pool, min_len, max_len, workers):
    """Endless deterministic stream of tasks (index order), optionally built by a process pool."""
    if workers <= 1:
        i = 0
        while True:
            yield make_task(seed, i, op_pool, min_len, max_len)
            i += 1
    with mp.Pool(workers) as pool:
        start, batch = 0, workers * 16
        while True:  # bounded batches keep memory flat and results in index order
            jobs = [(seed, i, op_pool, min_len, max_len) for i in range(start, start + batch)]
            start += batch
            for t in pool.map(_work, jobs, chunksize=4):
                if t is not None:
                    yield t


def generate(n: int, out: str, seed: int = 0, min_len=1, max_len=4, include=None, exclude=None,
             categories=None, trace=False, arc_files=True, log=print, workers=1) -> dict:
    op_pool = resolve_ops(include, exclude, categories)
    outdir = Path(out)
    (outdir / "tasks").mkdir(parents=True, exist_ok=True)
    seen, ops_used, lens = set(), Counter(), Counter()
    written, i = 0, 0
    with open(outdir / "dataset.jsonl", "w") as f:
        for task in _tasks(seed, op_pool, min_len, max_len, workers):
            if written >= n:
                break
            i += 1
            tid = task_id(task)
            if tid in seen:
                continue
            seen.add(tid)
            rec = record(task, tid, trace)
            f.write(json.dumps(rec) + "\n")
            if arc_files:
                (outdir / "tasks" / f"{tid}.json").write_text(json.dumps(arc_json(task)))
            ops_used.update(rec["ops"])
            lens[len(rec["ops"])] += 1
            written += 1
            if written % 100 == 0:
                log(f"{written}/{n}")
    stats = {"tasks": written, "seed": seed, "program_lengths": dict(sorted(lens.items())),
             "op_counts": dict(ops_used.most_common()),
             "unused_ops": sorted(set(op_pool) - set(ops_used))}
    (outdir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
