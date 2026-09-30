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


@op("crop_fixed", "geometry",
    lambda rng, g: {"corner": str(rng.choice(["tl", "tr", "bl", "br"])),
                    "h": int(rng.integers(1, max(2, g.shape[0] // 2 + 1))),
                    "w": int(rng.integers(1, max(2, g.shape[1] // 2 + 1)))})
def crop_fixed(g, corner, h, w):
    """Cut an h x w window out of one corner of the grid."""
    if h > g.shape[0] or w > g.shape[1]:
        raise OpError("too big")
    r = slice(0, h) if corner[0] == "t" else slice(g.shape[0] - h, None)
    c = slice(0, w) if corner[1] == "l" else slice(g.shape[1] - w, None)
    return g[r, c].copy()


@op("extract_period", "geometry", gens=("periodic",))
def extract_period(g):
    """Return the smallest tile whose repetition reproduces the whole grid."""
    from .relations import _find_period
    h, w = g.shape
    known = np.ones(g.shape, dtype=bool)
    best = None
    for py in range(1, h + 1):
        for px in range(1, w + 1):
            if (py, px) != (h, w) and h % py == 0 and w % px == 0 \
                    and (best is None or py * px < best[0]) \
                    and (g == np.tile(g[:py, :px], (h // py, w // px))).all():
                best = (py * px, py, px)
    if best is None:
        raise OpError("not periodic")
    return g[:best[1], :best[2]].copy()


def _parts(g, ny, nx):
    h, w = g.shape
    if ny * nx < 2 or h % ny or w % nx:
        raise OpError("cannot split evenly")
    a, b = h // ny, w // nx
    return [g[i * a:(i + 1) * a, j * b:(j + 1) * b] for i in range(ny) for j in range(nx)]


def _s_parts(rng, g):
    ny, nx = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)][int(rng.integers(5))]
    return {"ny": ny, "nx": nx}


@op("take_part", "geometry", lambda rng, g: {**_s_parts(rng, g), "idx": int(rng.integers(0, 4))}, gens=("parts",))
def take_part(g, ny, nx, idx):
    """Split into ny x nx equal parts (row-major) and return part idx."""
    parts = _parts(g, ny, nx)
    if idx >= len(parts):
        raise OpError("no such part")
    return parts[idx].copy()


@op("overlay_parts", "geometry", lambda rng, g: {**_s_parts(rng, g), "order": []}, gens=("parts",))
def overlay_parts(g, ny, nx, order):
    """Split into equal parts and stack them; the first part in `order` wins where several are non-empty."""
    parts = _parts(g, ny, nx)
    order = order or list(range(len(parts)))
    if sorted(order) != list(range(len(parts))):
        raise OpError("bad order")
    out = np.zeros_like(parts[0])
    for i in reversed(order):
        out = np.where(parts[i] != BG, parts[i], out)
    return out


@op("count_bar_fixed", "object",
    lambda rng, g: _os(rng, g, color=pick_color(rng), width=int(rng.integers(3, 10))))
def count_bar_fixed(g, sel, seg, color, width):
    """1 x width bar whose first N cells are coloured, N = number of selected objects."""
    n = len(_pick(g, seg, sel)[1])
    if n == 0 or n > width:
        raise OpError("bad count")
    out = np.zeros((1, width), dtype=int)
    out[0, :n] = color
    return out


@op("color_histogram", "relation", lambda rng, g: {"vertical": bool(rng.integers(0, 2))})
def color_histogram(g, vertical):
    """Bar chart of cell counts per foreground colour, most frequent first (columns or rows)."""
    vals, cnt = np.unique(g[g != BG], return_counts=True)
    if len(vals) < 2 or len(set(cnt.tolist())) != len(cnt) or cnt.max() > 30:
        raise OpError("need distinct counts")
    order = np.argsort(-cnt)
    out = np.zeros((len(vals), cnt.max()), dtype=int)
    for i, k in enumerate(order):
        out[i, :cnt[k]] = vals[k]
    return out.T[::-1].copy() if vertical else out
