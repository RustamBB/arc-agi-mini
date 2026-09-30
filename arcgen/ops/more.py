"""Ops found missing by beam-searching the real ARC training set (see tools/search_cov.py)."""
from __future__ import annotations

import numpy as np

from ..core import BG, OpError, find_objects, select
from .base import op, pick_color, pick_present, present, sample_sel, sample_seg
from .geometry import sym_complete
from .lines import RAYS
from .objects import DIRS, _os, _pick


@op("pad_replicate", "geometry", lambda rng, g: {"n": int(rng.integers(1, 3)), "corners": bool(rng.integers(0, 2))})
def pad_replicate(g, n, corners):
    """Surround the grid by repeating its edge cells outward (corners optionally left empty)."""
    out = np.pad(g, n, mode="edge")
    if corners:
        out[:n, :n] = out[:n, -n:] = out[-n:, :n] = out[-n:, -n:] = BG
    if max(out.shape) > 30:
        raise OpError("too large")
    return out


@op("symmetrize_bbox", "geometry",
    lambda rng, g: {"mode": str(rng.choice(["h", "v", "both", "rot", "d4"]))}, gens=("symbbox",))
def symmetrize_bbox(g, mode):
    """Complete a symmetric pattern about the centre of its own bounding box (not of the grid)."""
    r, c = np.nonzero(g != BG)
    if len(r) == 0:
        raise OpError("empty")
    r0, r1, c0, c1 = r.min(), r.max(), c.min(), c.max()
    out = g.copy()
    out[r0:r1 + 1, c0:c1 + 1] = sym_complete(g[r0:r1 + 1, c0:c1 + 1], mode)
    return out


@op("paint_interior", "object", lambda rng, g: _os(rng, g, color=pick_color(rng)))
def paint_interior(g, sel, seg, color):
    """Paint the interior (cells whose 8 neighbours all belong to the object) of selected objects."""
    out = g.copy()
    for o in _pick(g, seg, sel)[1]:
        m = np.pad(o.mask(), 1)
        inner = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                inner &= np.roll(m, (dy, dx), (0, 1))
        out[o.r0:o.r1 + 1, o.c0:o.c1 + 1][inner[1:-1, 1:-1]] = color
    return out


@op("recolor_split", "object",
    lambda rng, g: _os(rng, g, yes=pick_color(rng), no=pick_color(rng)))
def recolor_split(g, sel, seg, yes, no):
    """Binary classification of objects: those matching the selector get `yes`, all others `no`."""
    objs, chosen = _pick(g, seg, sel)
    ids = {id(o) for o in chosen}
    out = g.copy()
    for o in objs:
        out[o.rows, o.cols] = yes if id(o) in ids else no
    return out


@op("slide_cells", "object", lambda rng, g: {"src": pick_present(rng, g), "dir": str(rng.choice(list(DIRS)))})
def slide_cells(g, src, dir):
    """Cells of colour `src` move one by one in a direction until blocked by anything."""
    dy, dx = DIRS[dir]
    out = g.copy()
    h, w = g.shape
    cells = sorted(zip(*np.nonzero(g == src)), key=lambda p: -(p[0] * dy + p[1] * dx))
    for r, c in cells:
        while 0 <= r + dy < h and 0 <= c + dx < w and out[r + dy, c + dx] == BG:
            out[r + dy, c + dx], out[r, c] = src, BG
            r, c = r + dy, c + dx
    return out


@op("crop_to_color", "geometry",
    lambda rng, g: {"color": pick_present(rng, g), "inner": bool(rng.integers(0, 2))})
def crop_to_color(g, color, inner):
    """Crop to the bounding box of one colour (e.g. a marker frame); inner=True drops that frame ring."""
    r, c = np.nonzero(g == color)
    if len(r) == 0:
        raise OpError("colour absent")
    r0, r1, c0, c1 = r.min(), r.max(), c.min(), c.max()
    if inner:
        r0, r1, c0, c1 = r0 + 1, r1 - 1, c0 + 1, c1 - 1
        if r1 < r0 or c1 < c0:
            raise OpError("too thin")
    return g[r0:r1 + 1, c0:c1 + 1].copy()


@op("full_lines", "line",
    lambda rng, g: {"src": pick_present(rng, g), "dir": str(rng.choice(["h", "v", "cross", "diag", "star"]))},
    gens=("few", "sparse", "objects"))
def full_lines(g, src, dir):
    """Lines through every `src` cell across the whole grid, painting only empty cells (pass over obstacles)."""
    out = g.copy()
    h, w = g.shape
    for y, x in zip(*np.nonzero(g == src)):
        for dy, dx in RAYS[dir]:
            r, c = y + dy, x + dx
            while 0 <= r < h and 0 <= c < w:
                if out[r, c] == BG:
                    out[r, c] = src
                r, c = r + dy, c + dx
    return out


@op("connect_diag", "line",
    lambda rng, g: {"src": pick_present(rng, g), "line": -1 if rng.random() < .5 else pick_color(rng)},
    gens=("few",))
def connect_diag(g, src, line):
    """Join same-colour cells lying on a common diagonal with a diagonal line."""
    out = g.copy()
    lc = src if line < 0 else line
    ys, xs = np.nonzero(g == src)
    pts = list(zip(ys.tolist(), xs.tolist()))
    for (a, b) in pts:
        for (c, d) in pts:
            if c > a and abs(c - a) == abs(d - b):
                sx = 1 if d > b else -1
                between = [(a + k, b + sx * k) for k in range(1, c - a)]
                if all(g[y, x] == BG for y, x in between):
                    for y, x in between:
                        out[y, x] = lc
    return out
