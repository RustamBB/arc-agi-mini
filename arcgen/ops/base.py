"""Operation registry and parameter-sampling helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core import BG, find_objects, select

MAX_SEL_SIZE = 60  # keeps size thresholds inside the tokenizer's integer vocabulary
DEFAULT_GENS = ("objects", "sparse", "few", "noise", "rings")


@dataclass
class Op:
    name: str
    category: str
    fn: Callable
    sample: Callable  # (rng, probe_grid) -> params dict
    gens: tuple = DEFAULT_GENS
    doc: str = ""


OPS: dict[str, Op] = {}


def op(name: str, category: str, sample: Callable | None = None, gens: tuple | None = None):
    def deco(fn):
        assert name not in OPS, name
        OPS[name] = Op(name, category, fn, sample or (lambda rng, g: {}),
                       gens or DEFAULT_GENS, (fn.__doc__ or "").strip())
        return fn
    return deco


# ---- parameter sampling helpers -------------------------------------------------

def present(g) -> list[int]:
    return [int(c) for c in np.unique(g) if c != BG]


def pick_present(rng, g) -> int:
    p = present(g)
    return int(rng.choice(p)) if p else int(rng.integers(1, 10))


def pick_color(rng, exclude=()) -> int:
    cs = [c for c in range(1, 10) if c not in exclude]
    return int(rng.choice(cs))


def pick_new(rng, g) -> int:
    return pick_color(rng, present(g)) if len(present(g)) < 9 else pick_color(rng)


def sample_seg(rng) -> str:
    return str(rng.choice(["c8", "c4", "m8", "m4"], p=[.4, .25, .2, .15]))


def sample_sel(rng, g, seg) -> dict:
    """Pick a selector that picks a non-empty proper subset of the probe's objects."""
    objs = find_objects(g, seg)
    sizes = sorted({o.size for o in objs if o.size <= MAX_SEL_SIZE}) or [1]
    for _ in range(30):
        by = str(rng.choice(_SEL, p=_SEL_P))
        sel = {"by": by}
        if by in ("color", "not_color"):
            sel["value"] = pick_present(rng, g)
        elif by in ("size_gt", "size_lt", "size_eq"):
            sel["value"] = int(rng.choice(sizes))
        chosen = select(objs, sel, g.shape)
        if 0 < len(chosen) and (by == "all" or len(chosen) < len(objs)):
            return sel
    return {"by": "all"}


_SEL = ["largest", "smallest", "color", "not_color", "size_gt", "size_lt", "size_eq",
        "touches_border", "inner", "rect", "not_rect", "has_hole", "no_hole",
        "tallest", "widest", "all", "multicolor"]
_W = np.array([10, 10, 14, 4, 8, 8, 4, 6, 6, 5, 5, 6, 4, 3, 3, 3, 2], dtype=float)
_SEL_P = _W / _W.sum()
assert len(_SEL) == len(_SEL_P)
