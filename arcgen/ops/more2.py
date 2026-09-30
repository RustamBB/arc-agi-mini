"""Round-2 additions (see tools/search_cov.py near-miss analysis)."""
from __future__ import annotations

import numpy as np

from ..core import BG, MAX_SIZE, OpError, dilate_mask
from .base import op, pick_color, pick_present, present
from .objects import _os, _pick
from .relations import _two_colors

TFS = {
    "id": lambda g: g, "flip_h": lambda g: g[:, ::-1], "flip_v": lambda g: g[::-1],
    "rot180": lambda g: g[::-1, ::-1], "rot90": lambda g: np.rot90(g, 1),
    "rot270": lambda g: np.rot90(g, 3), "transpose": lambda g: g.T,
}


def _tf(g, name):
    return np.ascontiguousarray(TFS[name](g))


@op("react_touching", "relation",
    lambda rng, g: {**dict(zip(("a", "b"), _two_colors(rng, g))), "color": pick_color(rng)},
    gens=("markers", "attract", "few"))
def react_touching(g, a, b, color):
    """Colour-a cells that touch a colour-b cell become `color`; the touching b cells vanish."""
    A, B = g == a, g == b
    ta, tb = A & dilate_mask(B, 8), B & dilate_mask(A, 8)
    if not ta.any():
        raise OpError("no contact")
    out = g.copy()
    out[ta] = color
    out[tb] = BG
    return out


@op("recolor_mirrored", "relation", lambda rng, g: {"axis": int(rng.integers(0, 3)), "color": pick_color(rng)},
    gens=("symhalf", "sparse"))
def recolor_mirrored(g, axis, color):
    """Cells whose mirror image (0 up-down, 1 left-right, 2 both) has the same colour are recoloured."""
    m = {0: g[::-1], 1: g[:, ::-1], 2: g[::-1, ::-1]}[axis]
    return np.where((g != BG) & (g == m), color, g)


@op("keep_center_line", "geometry", lambda rng, g: {"axis": int(rng.integers(0, 2))}, gens=("odd",))
def keep_center_line(g, axis):
    """Keep only the middle column (axis 1) or middle row (axis 0) of an odd-sized grid."""
    n = g.shape[axis]
    if n % 2 == 0:
        raise OpError("even size")
    out = np.zeros_like(g)
    if axis == 1:
        out[:, n // 2] = g[:, n // 2]
    else:
        out[n // 2] = g[n // 2]
    return out


@op("crop_object_only", "object", lambda rng, g: _os(rng, g))
def crop_object_only(g, sel, seg):
    """Crop to the single selected object and drop everything else inside its box."""
    chosen = _pick(g, seg, sel)[1]
    if len(chosen) != 1:
        raise OpError("selection is not unique")
    return chosen[0].canvas()


@op("concat_with", "geometry",
    lambda rng, g: {"axis": int(rng.integers(0, 2)), "tf": str(rng.choice(list(TFS))), "swap": bool(rng.integers(0, 2))},
    gens=("small", "few", "objects"))
def concat_with(g, axis, tf, swap):
    """Glue the grid to a transformed copy of itself (side by side or stacked)."""
    t = _tf(g, tf)
    a, b = (t, g) if swap else (g, t)
    if a.shape[1 - axis] != b.shape[1 - axis]:
        raise OpError("shape mismatch")
    out = np.concatenate([a, b], axis=axis)
    if max(out.shape) > MAX_SIZE:
        raise OpError("too large")
    return out


@op("self_logic", "geometry",
    lambda rng, g: {"tf": str(rng.choice([t for t in TFS if t != "id"])),
                    "mode": str(rng.choice(["and", "or", "xor", "a_not_b"])), "color": pick_color(rng)},
    gens=("symhalf", "sparse"))
def self_logic(g, tf, mode, color):
    """Boolean combination of the grid's mask with a transformed copy of its mask."""
    t = _tf(g, tf)
    if t.shape != g.shape:
        raise OpError("shape mismatch")
    A, B = g != BG, t != BG
    m = {"and": A & B, "or": A | B, "xor": A ^ B, "a_not_b": A & ~B}[mode]
    return np.where(m, color, BG)


@op("recolor_by_frequency_rank", "color",
    lambda rng, g: {"colors": [pick_color(rng) for _ in range(int(rng.integers(2, 4)))]})
def recolor_by_frequency_rank(g, colors):
    """Most frequent foreground colour becomes colors[0], next colors[1], ... (others unchanged)."""
    vals, cnt = np.unique(g[g != BG], return_counts=True)
    if len(vals) < 2 or len(set(cnt.tolist())) != len(cnt):
        raise OpError("need distinct counts")
    out = g.copy()
    for rank, k in enumerate(np.argsort(-cnt)):
        if rank < len(colors):
            out[g == vals[k]] = colors[rank]
    return out


@op("swap_extreme_colors", "color")
def swap_extreme_colors(g):
    """Exchange the most and the least frequent foreground colours."""
    vals, cnt = np.unique(g[g != BG], return_counts=True)
    if len(vals) < 2 or cnt.max() == cnt.min() or (cnt == cnt.max()).sum() > 1 or (cnt == cnt.min()).sum() > 1:
        raise OpError("no unique extremes")
    hi, lo = vals[np.argmax(cnt)], vals[np.argmin(cnt)]
    return np.where(g == hi, lo, np.where(g == lo, hi, g))


@op("bbox_frame_all", "object", lambda rng, g: {"color": pick_color(rng), "pad": int(rng.integers(0, 3))})
def bbox_frame_all(g, color, pad):
    """Draw a rectangle around the bounding box of all content, `pad` cells away (empty cells only)."""
    r, c = np.nonzero(g != BG)
    if len(r) == 0:
        raise OpError("empty")
    r0, r1, c0, c1 = r.min() - pad, r.max() + pad, c.min() - pad, c.max() + pad
    out = g.copy()
    for i in range(r0, r1 + 1):
        for j in range(c0, c1 + 1):
            if (i in (r0, r1) or j in (c0, c1)) and 0 <= i < g.shape[0] and 0 <= j < g.shape[1] \
                    and g[i, j] == BG:
                out[i, j] = color
    return out


@op("majority_filter", "object", lambda rng, g: {"conn": int(rng.choice([4, 8]))}, gens=("blocky_noise", "objects"))
def majority_filter(g, conn):
    """Every cell takes the most common value of its neighbourhood (ties keep the cell): removes speckle."""
    h, w = g.shape
    nb = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn == 8 else [])
    out = g.copy()
    for i in range(h):
        for j in range(w):
            cnt = np.zeros(10, dtype=int)
            for dy, dx in nb:
                a, b = i + dy, j + dx
                if 0 <= a < h and 0 <= b < w:
                    cnt[g[a, b]] += 1
            top = cnt.max()
            if cnt[g[i, j]] < top and (cnt == top).sum() == 1:
                out[i, j] = int(cnt.argmax())
    return out


@op("mark_centers", "object", lambda rng, g: _os(rng, g, color=pick_color(rng)))
def mark_centers(g, sel, seg, color):
    """Put a `color` dot on the centre cell of every selected object whose box has odd height and width."""
    out = g.copy()
    for o in _pick(g, seg, sel)[1]:
        if o.h % 2 and o.w % 2:
            out[o.r0 + o.h // 2, o.c0 + o.w // 2] = color
    return out
