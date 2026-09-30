"""Object-layer operations: decompose into objects, edit a selection, recompose."""
from __future__ import annotations

import numpy as np

from ..core import BG, MAX_SIZE, N4, N8, OpError, dilate_mask, find_objects, label, select
from .base import op, pick_color, sample_sel, sample_seg

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _pick(g, seg, sel):
    objs = find_objects(g, seg)
    return objs, select(objs, sel, g.shape, g)


def _os(rng, g, **extra):
    seg = sample_seg(rng)
    return {"sel": sample_sel(rng, g, seg), "seg": seg, **extra}


def _erase(out, objs):
    for o in objs:
        out[o.rows, o.cols] = BG


def _paint(out, o, dy, dx):
    r, c = o.rows + dy, o.cols + dx
    if r.min() < 0 or c.min() < 0 or r.max() >= out.shape[0] or c.max() >= out.shape[1]:
        raise OpError("out of bounds")
    out[r, c] = o.vals


@op("recolor_objects", "object", lambda rng, g: _os(rng, g, color=pick_color(rng)))
def recolor_objects(g, sel, seg, color):
    """Paint the selected objects with one colour."""
    out = g.copy()
    for o in _pick(g, seg, sel)[1]:
        out[o.rows, o.cols] = color
    return out


def _s_rank(rng, g):
    return {"colors": [pick_color(rng) for _ in range(int(rng.integers(2, 4)))],
            "seg": sample_seg(rng)}


@op("recolor_by_size_rank", "object", _s_rank)
def recolor_by_size_rank(g, colors, seg):
    """Largest object gets colors[0], next size colors[1], ... (others unchanged)."""
    objs = find_objects(g, seg)
    sizes = sorted({o.size for o in objs}, reverse=True)
    out = g.copy()
    for o in objs:
        rank = sizes.index(o.size)
        if rank < len(colors):
            out[o.rows, o.cols] = colors[rank]
    return out


def _s_holes(rng, g):
    return {"colors": [pick_color(rng) for _ in range(3)], "seg": sample_seg(rng)}


@op("recolor_by_holes", "object", _s_holes, gens=("rings",))
def recolor_by_holes(g, colors, seg):
    """Colour objects by their number of enclosed holes (0, 1, 2+)."""
    out = g.copy()
    for o in find_objects(g, seg):
        out[o.rows, o.cols] = colors[min(o.n_holes(), 2)]
    return out


@op("delete_objects", "object", lambda rng, g: _os(rng, g))
def delete_objects(g, sel, seg):
    """Remove the selected objects."""
    out = g.copy()
    _erase(out, _pick(g, seg, sel)[1])
    return out


@op("keep_objects", "object", lambda rng, g: _os(rng, g))
def keep_objects(g, sel, seg):
    """Keep only the selected objects."""
    out = np.full_like(g, BG)
    for o in _pick(g, seg, sel)[1]:
        out[o.rows, o.cols] = o.vals
    return out


def _s_move(rng, g):
    while True:
        dy, dx = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
        if dy or dx:
            return _os(rng, g, dy=dy, dx=dx)


@op("move_objects", "object", _s_move)
def move_objects(g, sel, seg, dy, dx):
    """Translate the selected objects by (dy, dx)."""
    chosen = _pick(g, seg, sel)[1]
    out = g.copy()
    _erase(out, chosen)
    for o in chosen:
        _paint(out, o, dy, dx)
    return out


@op("copy_objects", "object", _s_move)
def copy_objects(g, sel, seg, dy, dx):
    """Stamp a copy of the selected objects at offset (dy, dx)."""
    out = g.copy()
    for o in _pick(g, seg, sel)[1]:
        r, c = o.rows + dy, o.cols + dx
        ok = (r >= 0) & (c >= 0) & (r < g.shape[0]) & (c < g.shape[1])
        out[r[ok], c[ok]] = o.vals[ok]
    return out


@op("slide_objects", "object",
    lambda rng, g: _os(rng, g, dir=str(rng.choice(list(DIRS)))))
def slide_objects(g, sel, seg, dir):
    """Slide selected objects rigidly until they hit the border or another object."""
    dy, dx = DIRS[dir]
    chosen = _pick(g, seg, sel)[1]
    out = g.copy()
    _erase(out, chosen)
    occ = out != BG
    for o in sorted(chosen, key=lambda o: -int((o.rows * dy + o.cols * dx).max())):
        r, c = o.rows.copy(), o.cols.copy()
        while True:
            nr, nc = r + dy, c + dx
            if nr.min() < 0 or nc.min() < 0 or nr.max() >= g.shape[0] or nc.max() >= g.shape[1] \
                    or occ[nr, nc].any():
                break
            r, c = nr, nc
        out[r, c] = o.vals
        occ[r, c] = True
    return out


@op("flip_objects", "object", lambda rng, g: _os(rng, g, axis=int(rng.integers(0, 2))))
def flip_objects(g, sel, seg, axis):
    """Mirror each selected object inside its own bounding box."""
    chosen = _pick(g, seg, sel)[1]
    out = g.copy()
    _erase(out, chosen)
    for o in chosen:
        cv = np.flip(o.canvas(), axis)
        sub = out[o.r0:o.r1 + 1, o.c0:o.c1 + 1]
        sub[cv != 0] = cv[cv != 0]
    return out


@op("fill_holes", "object", lambda rng, g: {"color": pick_color(rng)}, gens=("rings", "objects"))
def fill_holes(g, color):
    """Fill background regions that are fully enclosed (not connected to the border)."""
    lab, n = label(g == BG, 4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    enclosed = (g == BG) & ~np.isin(lab, list(border))
    return np.where(enclosed, color, g)


def _own_or(rng):
    return -1 if rng.random() < .3 else pick_color(rng)  # -1: the object's own colour


@op("outline_objects", "object", lambda rng, g: _os(rng, g, color=_own_or(rng)))
def outline_objects(g, sel, seg, color):
    """Draw a one-cell halo around the selected objects (colour -1: their own)."""
    out = g.copy()
    for o in _pick(g, seg, sel)[1]:
        m = np.zeros(g.shape, dtype=bool)
        m[o.rows, o.cols] = True
        out[dilate_mask(m, 8) & (g == BG)] = o.color if color < 0 else color
    return out


@op("bbox_fill", "object", lambda rng, g: _os(rng, g, color=pick_color(rng)))
def bbox_fill(g, sel, seg, color):
    """Fill the empty cells inside each selected object's bounding box."""
    out = g.copy()
    for o in _pick(g, seg, sel)[1]:
        sub = out[o.r0:o.r1 + 1, o.c0:o.c1 + 1]
        sub[sub == BG] = color
    return out


@op("frame_objects", "object", lambda rng, g: _os(rng, g, color=_own_or(rng)))
def frame_objects(g, sel, seg, color):
    """Draw a rectangular frame one cell outside each selected object's bounding box."""
    out = g.copy()
    h, w = g.shape
    for o in _pick(g, seg, sel)[1]:
        for r in range(o.r0 - 1, o.r1 + 2):
            for c in range(o.c0 - 1, o.c1 + 2):
                edge = r in (o.r0 - 1, o.r1 + 1) or c in (o.c0 - 1, o.c1 + 1)
                if edge and 0 <= r < h and 0 <= c < w and g[r, c] == BG:
                    out[r, c] = o.color if color < 0 else color
    return out


@op("hollow_objects", "object", lambda rng, g: _os(rng, g))
def hollow_objects(g, sel, seg):
    """Remove the interior of selected objects, leaving only their contour."""
    out = g.copy()
    for o in _pick(g, seg, sel)[1]:
        m = np.pad(o.mask(), 1)
        inner = m.copy()
        for dy, dx in N4:
            inner &= np.roll(m, (dy, dx), (0, 1))
        inner = inner[1:-1, 1:-1]
        out[o.r0:o.r1 + 1, o.c0:o.c1 + 1][inner] = BG
    return out


@op("dilate", "object", lambda rng, g: {"conn": int(rng.choice([4, 8]))})
def dilate(g, conn):
    """Grow every object by one cell."""
    out = g.copy()
    for dy, dx in (N8 if conn == 8 else N4):
        sh = np.zeros_like(g)
        h, w = g.shape
        sh[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
            g[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
        m = (out == BG) & (sh != BG)
        out[m] = sh[m]
    return out


@op("erode", "object", lambda rng, g: {"conn": int(rng.choice([4, 8]))})
def erode(g, conn):
    """Peel one layer of cells off every object."""
    fg = np.pad(g != BG, 1)
    keep = fg.copy()
    for dy, dx in (N8 if conn == 8 else N4):
        keep &= np.roll(fg, (dy, dx), (0, 1))
    keep = keep[1:-1, 1:-1]
    return np.where(keep, g, BG)


@op("crop_to_object", "object", lambda rng, g: _os(rng, g))
def crop_to_object(g, sel, seg):
    """Crop the grid to the bounding box of the (single) selected object."""
    chosen = _pick(g, seg, sel)[1]
    if len(chosen) != 1:
        raise OpError("selection is not unique")
    o = chosen[0]
    return g[o.r0:o.r1 + 1, o.c0:o.c1 + 1].copy()


@op("count_objects_bar", "object", lambda rng, g: _os(rng, g, color=pick_color(rng)))
def count_objects_bar(g, sel, seg, color):
    """Output a 1xN bar where N is the number of selected objects."""
    n = len(_pick(g, seg, sel)[1])
    if n == 0 or n > MAX_SIZE:
        raise OpError("bad count")
    return np.full((1, n), color)


def _down(g):
    out = np.full_like(g, BG)
    for j in range(g.shape[1]):
        col = g[:, j][g[:, j] != BG]
        if len(col):
            out[-len(col):, j] = col
    return out


@op("gravity_cells", "object", lambda rng, g: {"dir": str(rng.choice(list(DIRS)))})
def gravity_cells(g, dir):
    """Every cell falls in a direction until it stacks against the wall."""
    if dir == "down":
        return _down(g)
    if dir == "up":
        return _down(g[::-1])[::-1]
    if dir == "right":
        return _down(g.T).T
    return _down(g.T[::-1])[::-1].T


@op("sort_columns_by_height", "object", lambda rng, g: {"order": str(rng.choice(["asc", "desc"]))},
    gens=("bars", "sparse"))
def sort_columns_by_height(g, order):
    """Stack cells at the bottom of each column and sort the columns by height."""
    d = _down(g)
    cnt = (d != BG).sum(0)
    idx = np.argsort(cnt if order == "asc" else -cnt, kind="stable")
    return d[:, idx]
