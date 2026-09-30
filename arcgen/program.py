"""Programs (op sequences): execution, sampling, task construction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import MAX_SIZE, OpError
from .generators import GENERATORS
from .ops import OPS

Grid = np.ndarray


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def apply_step(g: Grid, name: str, params: dict) -> Grid:
    out = np.asarray(OPS[name].fn(g, **params))
    if out.ndim != 2 or not (1 <= out.shape[0] <= MAX_SIZE and 1 <= out.shape[1] <= MAX_SIZE):
        raise OpError("bad output shape")
    return out.astype(int)


def run(program: list[dict], g: Grid) -> list[Grid]:
    """Execute a program; returns [input, after step 1, ..., output]."""
    trace = [g]
    for s in program:
        trace.append(apply_step(trace[-1], s["op"], s["params"]))
    return trace


def same(a: Grid, b: Grid) -> bool:
    return a.shape == b.shape and bool((a == b).all())


def to_text(program: list[dict]) -> str:
    def fmt(v):
        if isinstance(v, dict):
            return "{" + ",".join(f"{k}={fmt(x)}" for k, x in v.items()) + "}"
        if isinstance(v, list):
            return "[" + ",".join(fmt(x) for x in v) + "]"
        return str(v).lower() if isinstance(v, bool) else str(v)
    return " | ".join(f"{s['op']}({','.join(f'{k}={fmt(v)}' for k, v in s['params'].items())})"
                      for s in program)


@dataclass
class Task:
    program: list[dict]
    train: list[tuple[Grid, Grid]]
    test: list[tuple[Grid, Grid]]
    traces: list[list[Grid]]
    gen: str


def _rand_grid(rng, gen, pal, hint, lo, hi):
    h, w = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
    return GENERATORS[gen](rng, pal, h, w, hint)


def sample_program(rng, length, op_pool, pal, lo, hi, max_tries=60):
    """Sample ops one by one against a probe grid so parameters fit real data."""
    for _ in range(max_tries):
        first = OPS[str(rng.choice(op_pool))]
        gen = str(rng.choice(first.gens))
        probe, prog = None, []
        cur_op = first
        ok = True
        for i in range(length):
            if i:
                cands = [n for n in op_pool if n != prog[-1]["op"]]
                cur_op = OPS[str(rng.choice(cands))]
            for _ in range(15):
                if i == 0:
                    probe = _rand_grid(rng, gen, pal, {}, lo, hi)
                try:
                    params = _plain(cur_op.sample(rng, probe))
                    if i == 0:  # regenerate the probe with the real hint (e.g. blocky k)
                        probe = _rand_grid(rng, gen, pal, params, lo, hi)
                    nxt = apply_step(probe, cur_op.name, params)
                except (OpError, ValueError, IndexError):
                    continue
                if same(nxt, probe):
                    continue
                prog.append({"op": cur_op.name, "params": params})
                probe = nxt
                break
            else:
                ok = False
                break
        if ok:
            return prog, gen
    return None, None


def build_task(rng, program, gen, pal, lo, hi, n_train, n_test, max_tries=60) -> Task | None:
    hint = program[0]["params"]
    pairs, traces, seen = [], [], set()
    for _ in range((n_train + n_test) * max_tries // 6 + max_tries):
        if len(pairs) == n_train + n_test:
            break
        g = _rand_grid(rng, gen, pal, hint, lo, hi)
        try:
            tr = run(program, g)
        except (OpError, ValueError, IndexError):
            continue
        out = tr[-1]
        key = g.tobytes() + bytes(g.shape)
        if key in seen or same(g, out) or not (out != 0).any() or not (g != 0).any():
            continue
        seen.add(key)
        pairs.append((g, out))
        traces.append(tr)
    if len(pairs) < n_train + n_test:
        return None
    # every step must matter on most pairs, and outputs must not all coincide
    for k in range(len(program)):
        eff = sum(not same(t[k], t[k + 1]) for t in traces)
        if eff * 2 < len(traces):
            return None
    if len({(o.tobytes(), o.shape) for _, o in pairs}) < 2:
        return None
    return Task(program, pairs[:n_train], pairs[n_train:], traces, gen)
