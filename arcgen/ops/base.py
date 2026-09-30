"""Operation registry and parameter-sampling helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core import BG, find_objects, select

MAX_SEL_SIZE = 60  # keeps size thresholds inside the tokenizer's integer vocabulary
DEFAULT_GENS = ("objects", "sparse", "few", "noise", "rings", "shapes")


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
        if by in ("color", "not_color", "touching"):
            sel["value"] = pick_present(rng, g)
        elif by == "holes_eq":
            sel["value"] = int(rng.integers(0, 3))
        elif by in ("size_gt", "size_lt", "size_eq"):
            sel["value"] = int(rng.choice(sizes))
        chosen = select(objs, sel, g.shape, g)
        if 0 < len(chosen) and (by == "all" or len(chosen) < len(objs)):
            return sel
    return {"by": "all"}


_SEL_W = {
    "largest": 10, "smallest": 10, "color": 14, "not_color": 4, "size_gt": 8, "size_lt": 8,
    "size_eq": 4, "touches_border": 6, "inner": 6, "rect": 5, "not_rect": 5, "has_hole": 6,
    "no_hole": 4, "tallest": 3, "widest": 3, "all": 3, "multicolor": 2,
    # relational / contextual selectors
    "leftmost": 4, "rightmost": 4, "topmost": 4, "bottommost": 4, "common_color": 3,
    "rare_color": 3, "touching": 6, "square": 2, "symmetric": 2, "dup_shape": 4, "unique_shape": 4,
    "common_shape": 3, "holes_eq": 3,
}
_SEL = list(_SEL_W)
_SEL_P = np.array(list(_SEL_W.values()), dtype=float)
_SEL_P /= _SEL_P.sum()


def enumerate_selectors(g, seg) -> list[dict]:
    """Every selector that picks a non-empty subset of ``g``'s objects (search uses this)."""
    objs = find_objects(g, seg)
    if not objs:
        return []
    cols = sorted({o.color for o in objs})
    sizes = sorted({o.size for o in objs if o.size <= MAX_SEL_SIZE})
    cands = [{"by": b} for b in _SEL if b not in ("color", "not_color", "size_gt", "size_lt", "size_eq",
                                                  "touching")]
    cands += [{"by": b, "value": c} for b in ("color", "not_color", "touching") for c in cols]
    cands += [{"by": b, "value": s} for b in ("size_eq", "size_gt", "size_lt") for s in sizes]
    cands += [{"by": "holes_eq", "value": k} for k in range(3)]
    out, seen = [], set()
    for sel in cands:
        ch = select(objs, sel, g.shape, g)
        if ch:
            key = tuple(id(o) for o in ch)
            if key not in seen or sel["by"] == "all":
                seen.add(key)
                out.append(sel)
    return out
