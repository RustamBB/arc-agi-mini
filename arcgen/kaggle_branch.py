"""Verified-prediction branch for the Kaggle ARC pipeline: operation-bank beam search with fail-closed gates.

Interface mirrors the notebook's other branches: ``verified_arcgen_predictions(task, time_limit)`` returns
``(predictions, audit)``; ``audit["accepted"]`` is True only if ALL of these hold (otherwise predictions == []):

1. a program of the operation bank reproduces every demonstration exactly (beam search; not exhaustive);
2. leave-one-demo-out: for every demo, searching again on the other demos *with the same op skeleton* finds a program
   that predicts the held-out demo exactly;
3. query consensus: programs found with different seeds (same skeleton) agree on every query output;
4. every query output is a valid ARC grid and differs from its input.

Precompute for a whole challenge file (run in the background while the GPU solver works)::

    python -m arcgen.kaggle_branch --challenges arc-agi_evaluation_challenges.json --out arcgen_predictions.jsonl \
        --workers 2 --seconds 45 [--model arcgen_gpt.pt]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

from .guided import skeleton
from .program import run, same, to_text
from .search import search
from .serialize import check_program

VERSION = 1


def _grid(x):
    return np.asarray(x, dtype=np.int64)


def _valid(g) -> bool:
    return isinstance(g, np.ndarray) and g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30


def _skeleton_search(pairs, skel, budget, seed):
    prog, _, _ = search(pairs, beam=4, tries=120, top=24, budget=budget, seed=seed,
                        ops_by_depth=[[n] for n in skel])
    return prog if prog and check_program(to_text(prog), pairs) else None


def _model_programs(model_bundle, pairs, query, budget_left, n=24):
    """Skeleton-restricted search over the model's most frequent op sequences (optional)."""
    from collections import Counter
    from .model import predict_programs
    model, enc = model_bundle
    texts, _ = predict_programs(model, enc, pairs, query, n, 0.9)
    skels = Counter(s for s in map(skeleton, texts or []) if s)
    return [s for s, _ in skels.most_common(8)]


def verified_arcgen_predictions(task: dict, *, time_limit: float = 30.0, model_bundle=None, seed: int = 0):
    t0 = time.perf_counter()
    audit = {"accepted": False, "version": VERSION}
    pairs = [(_grid(p["input"]), _grid(p["output"])) for p in task.get("train", [])]
    queries = [_grid(p["input"]) for p in task.get("test", [])]
    if len(pairs) < 2 or not queries:
        audit["reason"] = "too_few_examples"
        return [], audit
    left = lambda: time_limit - (time.perf_counter() - t0)

    # 1. find a program
    program = None
    if model_bundle is not None:
        try:
            for k, sk in enumerate(_model_programs(model_bundle, pairs, queries[0], left())):
                if left() < time_limit * 0.6:
                    break
                program = _skeleton_search(pairs, sk, max(0.5, time_limit * 0.05), seed + k)
                if program:
                    audit["found_by"] = "model_skeleton"
                    break
        except Exception as exc:  # the optional model must never break the branch
            audit["model_error"] = repr(exc)
    if program is None:
        program, _, _ = search(pairs, depth=3, beam=4, tries=60, top=24, budget=max(1.0, time_limit * 0.5), seed=seed)
        if program and not check_program(to_text(program), pairs):
            program = None
        audit["found_by"] = "beam" if program else None
    if not program:
        audit["reason"] = "no_program"
        return [], audit
    skel = tuple(s["op"] for s in program)
    audit["program"] = to_text(program)

    # 2. leave-one-demo-out with the same skeleton
    fold_budget = max(0.5, min(left() * 0.5, time_limit * 0.35) / len(pairs))
    loo = []
    for k, held in enumerate(pairs):
        if left() <= 0:
            audit["reason"] = "time_budget"
            return [], audit
        rest = [p for i, p in enumerate(pairs) if i != k]
        fold_prog = _skeleton_search(rest, skel, fold_budget, seed + 10 + k)
        ok = False
        if fold_prog:
            try:
                ok = same(run(fold_prog, held[0])[-1], held[1])
            except Exception:
                ok = False
        loo.append(bool(ok))
        if not ok:
            audit["loo"] = loo
            audit["reason"] = "loo_failed"
            return [], audit
    audit["loo"] = loo

    # 3. query consensus across seeds
    outputs = []
    try:
        for q in queries:
            outputs.append(run(program, q)[-1])
    except Exception:
        audit["reason"] = "query_error"
        return [], audit
    alt_budget = max(0.3, min(left(), time_limit * 0.15) / 3)
    for k in range(3):
        if left() <= 0:
            break
        alt = _skeleton_search(pairs, skel, alt_budget, seed + 100 + k)
        if not alt:
            continue
        try:
            for q, ref in zip(queries, outputs):
                if not same(run(alt, q)[-1], ref):
                    audit["reason"] = "query_ambiguity"
                    return [], audit
        except Exception:
            audit["reason"] = "query_ambiguity"
            return [], audit

    # 4. sanity of the outputs
    if not all(_valid(o) for o in outputs):
        audit["reason"] = "invalid_output"
        return [], audit
    if any(o.shape == q.shape and (o == q).all() for o, q in zip(outputs, queries)):
        audit["reason"] = "query_unchanged"
        return [], audit
    audit.update({"accepted": True, "reason": "exact_loo_consensus", "elapsed": round(time.perf_counter() - t0, 3)})
    return outputs, audit


# ---- background precompute over a challenge file ---------------------------------------------------------

_MODEL = None


def _init_worker(model_path):
    global _MODEL
    try:
        os.nice(10)  # leave the CPU to the GPU solver's feeders
    except Exception:
        pass
    if model_path and Path(model_path).is_file():
        try:
            import torch
            torch.set_num_threads(1)
            from .model import Encoder, load
            _MODEL = (load(model_path, "cpu"), Encoder())
        except Exception:
            _MODEL = None


def _solve(item):
    tid, task, seconds = item
    try:
        preds, audit = verified_arcgen_predictions(task, time_limit=seconds, model_bundle=_MODEL, seed=zlib.crc32(tid.encode()) % 9973)
    except Exception as exc:
        preds, audit = [], {"accepted": False, "reason": "error", "error": repr(exc)}
    return {"task_id": tid, "accepted": bool(audit.get("accepted")), "audit": audit,
            "predictions": [np.asarray(p).tolist() for p in preds]}


def precompute(challenges: str, out: str, workers: int = 2, seconds: float = 45.0, model: str | None = None,
               deadline: float | None = None):
    import multiprocessing as mp
    data = json.load(open(challenges))
    done = set()
    if Path(out).is_file():
        done = {json.loads(l)["task_id"] for l in open(out) if l.strip()}
    items = [(tid, t, seconds) for tid, t in data.items() if tid not in done]
    items.sort(key=lambda it: sum(np.asarray(p["input"]).size for p in it[1]["train"]))  # cheap tasks first
    print(f"arcgen branch: {len(items)} tasks, {workers} workers, {seconds}s each", flush=True)
    with open(out, "a") as f, mp.Pool(workers, initializer=_init_worker, initargs=(model,)) as pool:
        for n, res in enumerate(pool.imap_unordered(_solve, items), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if deadline and time.time() > deadline:
                print("arcgen branch: deadline reached, stopping early", flush=True)
                pool.terminate()
                break
    ok = sum(1 for l in open(out) if json.loads(l)["accepted"])
    print(f"arcgen branch finished: {ok} accepted", flush=True)


def load_results(path: str) -> dict:
    """task_id -> {"accepted", "predictions": [ndarray per query], "audit"} (tolerates a partial file)."""
    res = {}
    if Path(path).is_file():
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            r["predictions"] = [np.asarray(p, dtype=np.int64) for p in r["predictions"]]
            res[r["task_id"]] = r
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--model", default=""); ap.add_argument("--deadline", type=float, default=0, help="unix time")
    a = ap.parse_args()
    precompute(a.challenges, a.out, a.workers, a.seconds, a.model or None, a.deadline or None)
