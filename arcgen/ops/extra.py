"""More building blocks: summarising, pooling, mirrored/rotated tilings, shape-matching."""
from __future__ import annotations

import numpy as np

from ..core import BG, MAX_SIZE, OpError, find_objects, select, shape_key
from .base import op, pick_color, pick_present, present, sample_sel, sample_seg
from .lines import _boxes
from .objects import _os, _pick


def _maj(cell):
    nz = cell[cell != BG]
    return int(np.bincount(nz).argmax()) if len(nz) else BG


@op("cells_to_pixels", "structure", gens=("gridded",))
def cells_to_pixels(g):
    """Summarise every sub-grid of a separator-split grid by its majority colour (empty -> background)."""
    boxes = _boxes(g)
    ny = len({b[0] for b in boxes})
    if len(boxes) % ny:
        raise OpError("irregular")
    px = [_maj(g[a:b, c:d]) for a, b, c, d in boxes]
    return np.array(px).reshape(ny, len(boxes) // ny)


@op("pool", "geometry", lambda rng, g: {"k": int(rng.integers(2, 5)),
                                        "mode": str(rng.choice(["any", "all"]))}, gens=("blocky_noise",))
def pool(g, k, mode):
    """Shrink by k: each k x k block becomes its majority colour ('any') or its colour only if uniform ('all')."""
    h, w = g.shape
    if h % k or w % k:
        raise OpError("not divisible")
    out = np.zeros((h // k, w // k), dtype=int)
    for i in range(h // k):
        for j in range(w // k):
            b = g[i * k:(i + 1) * k, j * k:(j + 1) * k]
            if mode == "any":
                out[i, j] = _maj(b)
            elif (b == b[0, 0]).all():
                out[i, j] = b[0, 0]
    return out


@op("dedupe_adjacent", "geometry", lambda rng, g: {"axis": int(rng.integers(0, 3))}, gens=("stretched",))
def dedupe_adjacent(g, axis):
    """Collapse runs of identical adjacent rows (axis 0), columns (1) or both (2)."""
    out = g
    if axis in (0, 2):
        keep = [0] + [i for i in range(1, out.shape[0]) if not (out[i] == out[i - 1]).all()]
        out = out[keep]
    if axis in (1, 2):
        keep = [0] + [j for j in range(1, out.shape[1]) if not (out[:, j] == out[:, j - 1]).all()]
        out = out[:, keep]
    return out.copy()


@op("tile_flip", "geometry", lambda rng, g: {"ny": int(rng.integers(1, 4)), "nx": int(rng.integers(1, 4))},
    gens=("small", "few"))
def tile_flip(g, ny, nx):
    """Tile ny x nx where every other tile is mirrored (left-right in odd columns, up-down in odd rows)."""
    if ny * nx < 2:
        raise OpError("noop")
    rows = []
    for i in range(ny):
        row = []
        for j in range(nx):
            t = g[:, ::-1] if j % 2 else g
            row.append(t[::-1] if i % 2 else t)
        rows.append(np.hstack(row))
    out = np.vstack(rows)
    if max(out.shape) > MAX_SIZE:
        raise OpError("too large")
    return out


@op("rot_quad", "geometry", lambda rng, g: {"cw": bool(rng.integers(0, 2))}, gens=("small", "few"))
def rot_quad(g, cw):
    """2x2 kaleidoscope of a square grid: it and its 90/180/270 degree rotations."""
    if g.shape[0] != g.shape[1] or g.shape[0] * 2 > MAX_SIZE:
        raise OpError("square only")
    k = -1 if cw else 1
    a, b, c, d = g, np.rot90(g, k), np.rot90(g, -k), np.rot90(g, 2)
    return np.vstack([np.hstack([a, b]), np.hstack([c, d])])


@op("shear", "geometry", lambda rng, g: {"step": int(rng.choice([-2, -1, 1, 2])), "axis": int(rng.integers(0, 2))})
def shear(g, step, axis):
    """Roll row i by i*step cells (axis 1) or column j by j*step (axis 0), wrapping around."""
    a = g if axis == 1 else g.T
    out = np.array([np.roll(r, i * step) for i, r in enumerate(a)])
    return out if axis == 1 else out.T.copy()


@op("scale_by_colors", "geometry", gens=("few", "sparse", "small"))
def scale_by_colors(g):
    """Enlarge every cell by the number of distinct foreground colours."""
    k = len(present(g))
    if k < 2 or max(g.shape) * k > MAX_SIZE:
        raise OpError("bad factor")
    return np.kron(g, np.ones((k, k), dtype=int))


@op("object_color_cell", "object", lambda rng, g: _os(rng, g))
def object_color_cell(g, sel, seg):
    """Output a 1x1 grid holding the colour of the selected object(s) (must agree)."""
    cs = {o.color for o in _pick(g, seg, sel)[1]}
    if len(cs) != 1:
        raise OpError("ambiguous")
    return np.array([[cs.pop()]])


@op("swap_object_colors", "object", lambda rng, g: {"seg": str(rng.choice(["m8", "m4"]))}, gens=("dotted",))
def swap_object_colors(g, seg):
    """In two-coloured objects exchange the two colours."""
    out = g.copy()
    for o in find_objects(g, seg):
        v = np.unique(o.vals)
        if len(v) == 2:
            out[o.rows, o.cols] = np.where(o.vals == v[0], v[1], v[0])
    return out


@op("rotate_objects", "object", lambda rng, g: _os(rng, g, k=int(rng.integers(1, 4))))
def rotate_objects(g, sel, seg, k):
    """Rotate selected objects with a square bounding box by k*90 degrees in place."""
    chosen = [o for o in _pick(g, seg, sel)[1] if o.h == o.w]
    out = g.copy()
    for o in chosen:
        out[o.rows, o.cols] = BG
    for o in chosen:
        cv = np.rot90(o.canvas(), k)
        sub = out[o.r0:o.r1 + 1, o.c0:o.c1 + 1]
        sub[cv != 0] = cv[cv != 0]
    return out


@op("recolor_by_shape_match", "relation", lambda rng, g: {"src": pick_present(rng, g)}, gens=("shapepairs",))
def recolor_by_shape_match(g, src):
    """Objects of colour `src` adopt the colour of another object that has exactly the same shape."""
    objs = find_objects(g, "c8")
    keys = {}
    for o in objs:
        if o.color != src:
            keys.setdefault(shape_key(o), set()).add(o.color)
    out = g.copy()
    for o in objs:
        if o.color == src and len(keys.get(shape_key(o), ())) == 1:
            out[o.rows, o.cols] = next(iter(keys[shape_key(o)]))
    return out
