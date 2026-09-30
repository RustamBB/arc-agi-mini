"""Target-guided beam search: find an op sequence that maps every train input to its output.

Used to (a) measure how much of a real ARC set the bank can express, (b) annotate real tasks
with programs, (c) rank unsolved tasks by residual error to see what the bank is missing.
"""
from __future__ import annotations

import time

import numpy as np

from .core import OpError
from .ops import OPS
from .ops.base import enumerate_selectors
from .program import _plain, apply_step, same

# parameter names that hold an *output* colour / a colour that exists in the grid
OUT_COLOR_KEYS = {"color", "dst", "yes", "no"}
IN_COLOR_KEYS = {"src", "a", "b", "mask", "seed", "marker", "mover", "target", "line", "touch"}


def score(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape == b.shape:
        return int((a != b).sum())
    sd = abs(a.shape[0] - b.shape[0]) + abs(a.shape[1] - b.shape[1])
    ha, hb = np.bincount(a.ravel(), minlength=10), np.bincount(b.ravel(), minlength=10)
    return 400 + 25 * sd + int(abs(ha - hb).sum()) // 2


def total(grids, targets) -> int:
    return sum(score(g, t) for g, t in zip(grids, targets))


def _guide(rng, params, out_colors, in_colors, g=None, sels=None):
    """Bias parameters towards what the data offers: colours of the target / current grid and
    selectors that really select something."""
    p = dict(params)
    if "sel" in p and g is not None and rng.random() < .9:
        seg = p.get("seg", "c8")
        if seg not in sels:
            sels[seg] = enumerate_selectors(g, seg)
        if sels[seg]:
            p["sel"] = dict(sels[seg][int(rng.integers(len(sels[seg])))])
    for k, v in p.items():
        if k in OUT_COLOR_KEYS and isinstance(v, int) and v >= 0 and out_colors and rng.random() < .7:
            p[k] = int(rng.choice(out_colors))
        elif k in IN_COLOR_KEYS and isinstance(v, int) and v >= 0 and in_colors and rng.random() < .3:
            p[k] = int(rng.choice(in_colors))
        elif k == "colors" and isinstance(v, list) and out_colors and rng.random() < .8:
            perm = [int(c) for c in rng.permutation(out_colors)]
            p[k] = [perm[i % len(perm)] for i in range(len(v))]
    return p


def search(pairs, depth=3, beam=4, tries=60, top=24, budget=30.0, seed=0, ops=None):
    """Returns (program or None, best_total_error, best_program)."""
    rng = np.random.default_rng(seed)
    ins = [np.asarray(a) for a, _ in pairs]
    tgt = [np.asarray(b) for _, b in pairs]
    t0 = time.time()
    names = list(ops or OPS)
    out_cols = [int(c) for c in np.unique(np.concatenate([t.ravel() for t in tgt])) if c != 0] \
        or [int(c) for c in np.unique(tgt[0])]
    keep_shape = all(a.shape == b.shape for a, b in zip(ins, tgt))
    frontier = [([], ins, total(ins, tgt))]
    best = frontier[0]
    if best[2] == 0:
        return [], 0, []
    seen = {ins[0].tobytes() + bytes(ins[0].shape)}
    for _ in range(depth):
        cands = []
        for prog, grids, cur in frontier:
            g0 = grids[0]
            in_cols = [int(c) for c in np.unique(g0)]
            s0_cur = score(g0, tgt[0])
            sels: dict = {}
            for name in names:
                if time.time() - t0 > budget:
                    break
                o = OPS[name]
                tried, dups = set(), 0
                for _ in range(tries):
                    try:
                        params = _guide(rng, _plain(o.sample(rng, g0)), out_cols, in_cols, g0, sels)
                        key = repr(sorted(params.items()))
                        if key in tried:
                            dups += 1
                            if dups > 12:
                                break
                            continue
                        tried.add(key)
                        n0 = apply_step(g0, name, params)
                    except (OpError, ValueError, IndexError, TypeError, KeyError, AssertionError):
                        continue
                    if same(n0, g0) or (keep_shape and n0.shape != g0.shape):
                        continue
                    s0 = score(n0, tgt[0])
                    cands.append((s0, s0_cur, name, params, prog, grids, n0))
        cands.sort(key=lambda c: c[0])
        nxt, dup = [], set()
        for s0, _, name, params, prog, grids, n0 in cands[:top * 3]:
            k = n0.tobytes() + bytes(n0.shape)
            if k in dup or k in seen:
                continue
            dup.add(k)
            try:
                new = [n0] + [apply_step(g, name, params) for g in grids[1:]]
            except (OpError, ValueError, IndexError, TypeError, KeyError, AssertionError):
                continue
            tot = total(new, tgt)
            step = {"op": name, "params": params}
            nxt.append((prog + [step], new, tot))
            if tot == 0:
                return prog + [step], 0, prog + [step]
            if len(nxt) >= top:
                break
        if not nxt:
            break
        seen |= dup
        nxt.sort(key=lambda s: s[2])
        frontier = nxt[:beam]
        if frontier[0][2] < best[2]:
            best = frontier[0]
        if time.time() - t0 > budget:
            break
    return None, best[2], best[0]
