"""Random input-grid generators. Every generator: ``gen(rng, pal, h, w, hint) -> grid``.

``pal`` is the task palette (colours used by every input of a task, so that
colour-dependent parameters stay meaningful) and ``hint`` holds the parameters
of the program's first step, for generators that must match one specific op.
"""
from __future__ import annotations

import numpy as np

from .core import BG


def _c(rng, pal):
    return int(rng.choice(pal))


def _rect(rng, pal, hollow=None):
    h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    a = np.full((h, w), _c(rng, pal))
    if hollow is None:
        hollow = rng.random() < .3
    if hollow and h > 2 and w > 2:
        a[1:-1, 1:-1] = BG
    return a


def _blob(rng, pal, multi=False):
    n = int(rng.integers(2, 9))
    cells = {(0, 0)}
    while len(cells) < n:
        y, x = list(cells)[int(rng.integers(len(cells)))]
        dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][int(rng.integers(4))]
        cells.add((y + dy, x + dx))
    ys, xs = zip(*cells)
    y0, x0 = min(ys), min(xs)
    a = np.zeros((max(ys) - y0 + 1, max(xs) - x0 + 1), dtype=int)
    col = _c(rng, pal)
    for y, x in cells:
        a[y - y0, x - x0] = _c(rng, pal) if multi else col
    return a


def _line(rng, pal):
    n = int(rng.integers(2, 7))
    a = np.full((1, n), _c(rng, pal))
    return a if rng.random() < .5 else a.T


def _shape(rng, pal):
    k = rng.choice(["rect", "hollow", "blob", "multi", "line", "dot"], p=[.22, .18, .28, .1, .12, .1])
    if k == "rect":
        return _rect(rng, pal, False)
    if k == "hollow":
        return _rect(rng, pal, True)
    if k == "blob":
        return _blob(rng, pal)
    if k == "multi":
        return _blob(rng, pal, True)
    if k == "line":
        return _line(rng, pal)
    return np.full((1, 1), _c(rng, pal))


def _place(rng, canvas, shape, gap=1, tries=40):
    H, W = canvas.shape
    h, w = shape.shape
    if h > H or w > W:
        return False
    for _ in range(tries):
        y, x = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
        y0, x0 = max(0, y - gap), max(0, x - gap)
        if (canvas[y0:y + h + gap, x0:x + w + gap] == BG).all():
            sub = canvas[y:y + h, x:x + w]
            sub[shape != 0] = shape[shape != 0]
            return True
    return False


def gen_objects(rng, pal, h, w, hint):
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(rng.integers(2, 8))):
        _place(rng, g, _shape(rng, pal), gap=int(rng.integers(0, 2)))
    return g


def gen_rings(rng, pal, h, w, hint):
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(rng.integers(2, 6))):
        a = _rect(rng, pal, True) if rng.random() < .8 else _shape(rng, pal)
        if a.shape[0] < 3 or a.shape[1] < 3:
            a = np.pad(a, 1, constant_values=_c(rng, pal))
            a[1:-1, 1:-1] = BG
        if rng.random() < .4:  # open the ring
            y, x = (0, int(rng.integers(1, a.shape[1] - 1))) if rng.random() < .5 \
                else (int(rng.integers(1, a.shape[0] - 1)), 0)
            a[y, x] = BG
        _place(rng, g, a, gap=1)
    return g


def gen_sparse(rng, pal, h, w, hint):
    m = rng.random((h, w)) < rng.uniform(.05, .22)
    return np.where(m, rng.choice(pal, size=(h, w)), BG)


def gen_noise(rng, pal, h, w, hint):
    m = rng.random((h, w)) < rng.uniform(.3, .7)
    return np.where(m, rng.choice(pal, size=(h, w)), BG)


def gen_few(rng, pal, h, w, hint):
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(rng.integers(2, 7))):
        g[int(rng.integers(h)), int(rng.integers(w))] = _c(rng, pal)
    return g


def gen_aligned(rng, pal, h, w, hint):
    """Pairs of same-coloured cells sharing a row or column, plus distractors."""
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(rng.integers(1, 5))):
        c = _c(rng, pal)
        if rng.random() < .5:
            y = int(rng.integers(h))
            a, b = rng.choice(w, 2, replace=False)
            g[y, a] = g[y, b] = c
        else:
            x = int(rng.integers(w))
            a, b = rng.choice(h, 2, replace=False)
            g[a, x] = g[b, x] = c
    return g


def gen_small(rng, pal, h, w, hint):
    h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    m = rng.random((h, w)) < .55
    g = np.where(m, rng.choice(pal, size=(h, w)), BG)
    if not g.any():
        g[0, 0] = _c(rng, pal)
    return g


def gen_blocky(rng, pal, h, w, hint):
    k = int(hint.get("k", 2))
    h, w = max(2, h // k), max(2, w // k)
    m = rng.random((h, w)) < .5
    g = np.where(m, rng.choice(pal, size=(h, w)), BG)
    return np.kron(g, np.ones((k, k), dtype=int))


def gen_bars(rng, pal, h, w, hint):
    g = np.zeros((h, w), dtype=int)
    col = _c(rng, pal)
    for j in range(w):
        n = int(rng.integers(0, h + 1))
        if n:
            g[h - n:, j] = col if rng.random() < .8 else _c(rng, pal)
    return g


def gen_gridded(rng, pal, h, w, hint):
    sep = _c(rng, pal)
    inner = [c for c in pal if c != sep] or [1 if sep != 1 else 2]
    ch, cw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    ny, nx = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    if ny * nx < 2:
        nx = 2
    H, W = ny * ch + ny - 1, nx * cw + nx - 1
    g = np.zeros((H, W), dtype=int)
    for i in range(1, ny):
        g[i * (ch + 1) - 1, :] = sep
    for j in range(1, nx):
        g[:, j * (cw + 1) - 1] = sep
    for i in range(ny):
        for j in range(nx):
            m = rng.random((ch, cw)) < rng.uniform(.1, .7)
            g[i * (ch + 1):i * (ch + 1) + ch, j * (cw + 1):j * (cw + 1) + cw] = \
                np.where(m, rng.choice(inner, size=(ch, cw)), BG)
    return g


def gen_halves(rng, pal, h, w, hint):
    axis = int(hint.get("axis", 1))
    n, m = int(rng.integers(3, 8)), int(rng.integers(3, 9))
    ca, cb, cs = [_c(rng, pal) for _ in range(3)]
    A = np.where(rng.random((m, n)) < .5, ca, BG)
    B = np.where(rng.random((m, n)) < .5, cb, BG)
    if axis == 1:
        return np.hstack([A, np.full((m, 1), cs), B])
    return np.vstack([A, np.full((1, n), cs), B])


def _orbit(mode, i, j, h, w):
    if mode == "h":
        return {(i, j), (i, w - 1 - j)}
    if mode == "v":
        return {(i, j), (h - 1 - i, j)}
    if mode == "both":
        return {(i, j), (i, w - 1 - j), (h - 1 - i, j), (h - 1 - i, w - 1 - j)}
    return {(i, j), (j, h - 1 - i), (h - 1 - i, h - 1 - j), (h - 1 - j, i)}  # rot


def gen_symhalf(rng, pal, h, w, hint):
    """A symmetric pattern with most cells of every symmetry orbit erased."""
    mode = hint.get("mode", "h")
    if mode == "rot":
        h = w = min(h, w)
    out = np.zeros((h, w), dtype=int)
    seen = set()
    for i in range(h):
        for j in range(w):
            if (i, j) in seen:
                continue
            members = sorted(_orbit(mode, i, j, h, w))
            seen.update(members)
            if rng.random() < .5:
                keep = members[int(rng.integers(len(members)))]
                col = _c(rng, pal)
                for cell in members:
                    if cell == keep or rng.random() < .2:
                        out[cell] = col
    return out


GENERATORS = {
    "objects": gen_objects, "rings": gen_rings, "sparse": gen_sparse, "noise": gen_noise,
    "few": gen_few, "aligned": gen_aligned, "small": gen_small, "blocky": gen_blocky,
    "bars": gen_bars, "gridded": gen_gridded, "halves": gen_halves, "symhalf": gen_symhalf,
}
