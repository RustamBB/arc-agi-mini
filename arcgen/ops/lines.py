"""Line drawing, structure (sub-grid) and stamp operations."""
from __future__ import annotations

import numpy as np

from ..core import BG, OpError
from .base import op, pick_color, pick_present, present

RAYS = {
    "up": [(-1, 0)], "down": [(1, 0)], "left": [(0, -1)], "right": [(0, 1)],
    "h": [(0, -1), (0, 1)], "v": [(-1, 0), (1, 0)],
    "cross": [(-1, 0), (1, 0), (0, -1), (0, 1)],
    "diag": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    "ul": [(-1, -1)], "ur": [(-1, 1)], "dl": [(1, -1)], "dr": [(1, 1)],
    "star": [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
}


def _s_rays(rng, g):
    src = pick_present(rng, g) if rng.random() < .7 else -1
    return {"src": src, "dir": str(rng.choice(list(RAYS)))}


@op("shoot_rays", "line", _s_rays, gens=("few", "sparse"))
def shoot_rays(g, src, dir):
    """Cells of colour src (-1: any) shoot rays over the background until an obstacle."""
    out = g.copy()
    h, w = g.shape
    ys, xs = np.nonzero(g == src) if src >= 0 else np.nonzero(g != BG)
    for y, x in zip(ys, xs):
        for dy, dx in RAYS[dir]:
            r, c = y + dy, x + dx
            while 0 <= r < h and 0 <= c < w and g[r, c] == BG:
                out[r, c] = g[y, x]
                r, c = r + dy, c + dx
    return out


@op("connect_pairs", "line",
    lambda rng, g: {"src": -1 if rng.random() < .3 else pick_present(rng, g),
                    "line": -1 if rng.random() < .5 else pick_color(rng)},
    gens=("aligned", "few"))
def connect_pairs(g, src, line):
    """Join same-colour cells that share a row or column with a straight line (src -1: any colour)."""
    out = g.copy()
    for axis in (0, 1):
        m = g if axis == 0 else g.T
        o = out if axis == 0 else out.T
        for i in range(m.shape[0]):
            xs = np.nonzero(m[i] != BG if src < 0 else m[i] == src)[0]
            for a, b in zip(xs[:-1], xs[1:]):
                if m[i, a] != m[i, b]:
                    continue
                seg = o[i, a + 1:b]
                seg[seg == BG] = m[i, a] if line < 0 else line
    return out


@op("fill_lines", "line",
    lambda rng, g: {"src": pick_present(rng, g), "axis": int(rng.integers(0, 3)),
                    "color": pick_color(rng)}, gens=("few", "sparse"))
def fill_lines(g, src, axis, color):
    """Paint the empty cells of every row (0), column (1) or both (2) containing src."""
    out = g.copy()
    ys, xs = np.nonzero(g == src)
    if axis in (0, 2):
        for y in set(ys.tolist()):
            out[y][g[y] == BG] = color
    if axis in (1, 2):
        for x in set(xs.tolist()):
            out[:, x][g[:, x] == BG] = color
    return out


PATTERNS = {
    "plus": [(-1, 0), (1, 0), (0, -1), (0, 1)],
    "x": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    "ring": [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    "row": [(0, -1), (0, 1)],
    "col": [(-1, 0), (1, 0)],
}


@op("stamp_shape", "line",
    lambda rng, g: {"src": pick_present(rng, g), "shape": str(rng.choice(list(PATTERNS))),
                    "color": -1 if rng.random() < .4 else pick_color(rng)},
    gens=("few", "sparse"))
def stamp_shape(g, src, shape, color):
    """Grow a small pattern (plus, x, ring...) around every cell of colour src."""
    out = g.copy()
    h, w = g.shape
    for y, x in zip(*np.nonzero(g == src)):
        for dy, dx in PATTERNS[shape]:
            r, c = y + dy, x + dx
            if 0 <= r < h and 0 <= c < w and g[r, c] == BG:
                out[r, c] = src if color < 0 else color
    return out


def _boxes(g):
    """(r0, r1, c0, c1) half-open boxes of the sub-grids of a separator-split grid, row-major."""
    h, w = g.shape
    sep = None
    for line in list(g) + list(g.T):
        if line[0] != BG and (line == line[0]).all():
            sep = int(line[0])
            break
    if sep is None:
        raise OpError("no separators")
    rs = [i for i in range(h) if (g[i] == sep).all()]
    cs = [j for j in range(w) if (g[:, j] == sep).all()]

    def spans(idx, n):
        out, prev = [], -1
        for i in idx + [n]:
            if i - prev > 1:
                out.append((prev + 1, i))
            prev = i
        return out

    boxes = [(a, b, c, d) for a, b in spans(rs, h) for c, d in spans(cs, w)]
    if len(boxes) < 2:
        raise OpError("single cell")
    return boxes


def _cells(g):
    return [g[a:b, c:d] for a, b, c, d in _boxes(g)]


@op("select_cell", "structure", lambda rng, g: {"mode": str(rng.choice(["most", "least"]))},
    gens=("gridded",))
def select_cell(g, mode):
    """In a grid split by separator lines pick the fullest / emptiest cell."""
    cells = _cells(g)
    cnt = np.array([(c != BG).sum() for c in cells])
    tgt = cnt.max() if mode == "most" else cnt.min()
    if (cnt == tgt).sum() != 1:
        raise OpError("not unique")
    return cells[int(np.argmax(cnt == tgt))].copy()


@op("overlay_cells", "structure", lambda rng, g: {"order": str(rng.choice(["fwd", "rev"]))},
    gens=("gridded",))
def overlay_cells(g, order):
    """Stack all sub-grids of a separator-split grid on top of each other."""
    cells = _cells(g)
    if len({c.shape for c in cells}) != 1:
        raise OpError("cells differ in shape")
    if order == "rev":
        cells = cells[::-1]
    out = cells[0].copy()
    for c in cells[1:]:
        out = np.where(out != BG, out, c)
    return out
