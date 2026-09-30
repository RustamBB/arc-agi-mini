"""Generators for the relational ops. Hints (first op's params) fix the colours/modes involved."""
from __future__ import annotations

import numpy as np

from .core import BG, label
from .generators import (GENERATORS, _blob, _c, _orbit, _place, _rect, _shape)


def _without(pal, *ex):
    r = [c for c in pal if c not in ex]
    return r or [c for c in range(1, 10) if c not in ex]


def gen_shapes(rng, pal, h, w, hint):
    """Repeated shapes (same / different colour) plus odd ones out."""
    g = np.zeros((h, w), dtype=int)
    same_color = rng.random() < .5
    for p in [_shape(rng, pal) for _ in range(int(rng.integers(1, 4)))]:
        for _ in range(int(rng.integers(1, 4))):
            s = p if same_color or rng.random() < .5 else np.where(p != 0, _c(rng, pal), 0)
            _place(rng, g, s, gap=1)
    return g


def gen_tiled(rng, pal, h, w, hint):
    mask = int(hint.get("mask", 0))
    cols = _without(pal, mask) + ([BG] if mask != BG and rng.random() < .3 else [])
    for _ in range(30):
        py, px = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        H, W = max(h, 2 * py + 1), max(w, 2 * px + 1)
        tile = rng.choice(cols, size=(py, px))
        if len(np.unique(tile)) < 2 and py * px > 1:
            continue
        g = np.tile(tile, (H // py + 1, W // px + 1))[:H, :W].copy()
        m = np.zeros_like(g, dtype=bool)
        for _ in range(int(rng.integers(1, 3))):
            a, b = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            y, x = int(rng.integers(0, H - a + 1)), int(rng.integers(0, W - b + 1))
            m[y:y + a, x:x + b] = True
        ii, jj = np.indices(g.shape)
        cls = (ii % py) * px + jj % px
        if len(np.unique(cls[~m])) == py * px:
            g[m] = mask
            return g
    return np.tile(np.array([[1, 2], [2, 1]]), (4, 4))


def gen_symmask(rng, pal, h, w, hint):
    mode, mask = hint.get("mode", "h"), int(hint.get("mask", 0))
    if mode == "rot":
        h = w = min(h, w)
    cols = _without(pal, mask)
    for _ in range(30):
        g = np.zeros((h, w), dtype=int)
        seen = set()
        for i in range(h):
            for j in range(w):
                if (i, j) not in seen:
                    mem = _orbit(mode, i, j, h, w)
                    seen |= mem
                    col = _c(rng, cols)
                    for cell in mem:
                        g[cell] = col
        a, b = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        y, x = int(rng.integers(0, h - a + 1)), int(rng.integers(0, w - b + 1))
        m = np.zeros((h, w), dtype=bool)
        m[y:y + a, x:x + b] = True
        if all(any(not m[c] for c in _orbit(mode, i, j, h, w)) for i, j in zip(*np.nonzero(m))):
            g[m] = mask
            return g
    return np.full((h, w), cols[0])


def gen_corners(rng, pal, h, w, hint):
    g = np.zeros((h, w), dtype=int)
    cs = list(rng.permutation(pal))[:int(rng.integers(1, 4))]
    for c in cs:
        for _ in range(20):
            (a, b), (d, e) = rng.integers(0, h, 2), rng.integers(0, w, 2)
            if a != b and d != e and g[a, d] == g[b, e] == 0:
                g[a, d] = g[b, e] = c
                break
    return g


def gen_prefixrows(rng, pal, h, w, hint):
    a = np.zeros((h, w), dtype=int)
    for i in range(h):
        if rng.random() < .6:
            p = int(rng.integers(1, 4))
            pat = rng.choice(list(pal) + [BG], size=p, p=[.8 / len(pal)] * len(pal) + [.2])
            if not pat.any():
                pat[0] = _c(rng, pal)
            n = int(rng.integers(2 * p, max(2 * p, w - 1) + 1))
            while n < w and pat[(n - 1) % p] == BG:
                n += 1
            if n < w:
                a[i, :n] = [pat[k % p] for k in range(n)]
    return a.T.copy() if hint.get("axis", 1) == 0 else a


def gen_dotted(rng, pal, h, w, hint):
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(rng.integers(2, 6))):
        s = _rect(rng, pal, False) if rng.random() < .5 else _blob(rng, pal)
        if (s != 0).sum() < 4:
            continue
        cells = list(zip(*np.nonzero(s)))
        other = _c(rng, _without(pal, int(s[cells[0]])))
        for k in rng.choice(len(cells), int(rng.integers(1, 3)), replace=False):
            s[cells[k]] = other
        _place(rng, g, s, gap=1)
    return g


def gen_lined(rng, pal, h, w, hint):
    line = int(hint.get("line", pal[0]))
    cols = _without(pal, line)
    g = np.where(rng.random((h, w)) < .12, rng.choice(cols, size=(h, w)), BG)
    if rng.random() < .5:
        g[int(rng.integers(2, max(3, h - 2)))] = line
    else:
        g[:, int(rng.integers(2, max(3, w - 2)))] = line
    return g


def gen_seeded(rng, pal, h, w, hint):
    seed = int(hint.get("seed", pal[-1]))
    cols = _without(pal, seed)
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(rng.integers(2, 6))):
        a = np.full((int(rng.integers(4, 8)), int(rng.integers(4, 8))), _c(rng, cols))
        a[1:-1, 1:-1] = BG
        if rng.random() < .65:
            a[int(rng.integers(1, a.shape[0] - 1)), int(rng.integers(1, a.shape[1] - 1))] = seed
        _place(rng, g, a, gap=1)
    return g


def gen_template(rng, pal, h, w, hint):
    marker = int(hint.get("marker", pal[-1]))
    cols = _without(pal, marker)
    for _ in range(50):
        t = (rng.random((3, 3)) < .6).astype(int)
        if t.sum() >= 3 and label(t > 0, 8)[1] == 1:
            break
    else:
        t = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    one = _c(rng, cols)
    t = np.where(t > 0, one if rng.random() < .6 else rng.choice(cols, size=t.shape), 0)
    g = np.zeros((h, w), dtype=int)
    _place(rng, g, t, gap=1)
    for _ in range(int(rng.integers(2, 5))):
        fp = np.zeros((3, 3), dtype=int)
        fp[1, 1] = marker
        _place(rng, g, fp, gap=1)
    return g


def gen_attract(rng, pal, h, w, hint):
    mover = int(hint.get("mover", pal[0]))
    target = int(hint.get("target", pal[1] if pal[1] != mover else pal[0]))
    g = np.zeros((h, w), dtype=int)
    side = int(rng.integers(4))
    t = int(rng.integers(1, 3))
    if side == 0:
        g[:t] = target
    elif side == 1:
        g[-t:] = target
    elif side == 2:
        g[:, :t] = target
    else:
        g[:, -t:] = target
    for _ in range(int(rng.integers(1, 4))):
        s = _blob(rng, [mover]) if rng.random() < .6 else np.full((1, 1), mover)
        _place(rng, g, s, gap=1)
    return g


def gen_markers(rng, pal, h, w, hint):
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(rng.integers(2, 6))):
        body = _rect(rng, pal, False) if rng.random() < .5 else _blob(rng, pal)
        if body.size < 2 or (body != 0).sum() < 2:
            continue
        bc = int(body[body != 0][0])
        padded = np.pad(body, 1)
        if rng.random() < .75:
            cand = [(i, j) for i in range(padded.shape[0]) for j in range(padded.shape[1])
                    if padded[i, j] == 0 and any(0 <= i + a < padded.shape[0] and 0 <= j + b < padded.shape[1]
                                                 and padded[i + a, j + b] for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1)))]
            i, j = cand[int(rng.integers(len(cand)))]
            padded[i, j] = _c(rng, _without(pal, bc))
        _place(rng, g, padded, gap=0)
    return g


def gen_containers(rng, pal, h, w, hint):
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(rng.integers(1, 4))):
        a, b = int(rng.integers(5, 9)), int(rng.integers(5, 9))
        col = _c(rng, pal)
        box = np.full((a, b), col)
        box[1:-1, 1:-1] = BG
        inner = np.zeros((a - 4, b - 4), dtype=int)
        for _ in range(int(rng.integers(1, 3))):
            _place(rng, inner, np.full((1, 1), _c(rng, _without(pal, col))), gap=1)
        box[2:-2, 2:-2] = inner
        _place(rng, g, box, gap=1)
    for _ in range(int(rng.integers(0, 3))):
        _place(rng, g, np.full((1, 1), _c(rng, pal)), gap=1)
    return g


GENERATORS.update({
    "shapes": gen_shapes, "tiled": gen_tiled, "symmask": gen_symmask, "corners": gen_corners,
    "prefixrows": gen_prefixrows, "dotted": gen_dotted, "lined": gen_lined, "seeded": gen_seeded,
    "template": gen_template, "attract": gen_attract, "markers": gen_markers,
    "containers": gen_containers,
})


def gen_blocky_noise(rng, pal, h, w, hint):
    k = int(hint.get("k", 2))
    a, b = max(2, h // k), max(2, w // k)
    g = np.kron(np.where(rng.random((a, b)) < .5, rng.choice(pal, size=(a, b)), BG), np.ones((k, k), dtype=int))
    m = rng.random(g.shape) < .1
    return np.where(m, rng.choice(pal, size=g.shape), g)


def gen_stretched(rng, pal, h, w, hint):
    shape = (int(rng.integers(2, 6)), int(rng.integers(2, 6)))
    base = np.where(rng.random(shape) < .7, rng.choice(pal, size=shape), BG)
    rows = np.repeat(np.arange(base.shape[0]), rng.integers(1, 4, base.shape[0]))
    cols = np.repeat(np.arange(base.shape[1]), rng.integers(1, 4, base.shape[1]))
    return base[np.ix_(rows, cols)]


def gen_shapepairs(rng, pal, h, w, hint):
    src = int(hint.get("src", pal[0]))
    others = _without(pal, src)
    g = np.zeros((h, w), dtype=int)
    for _ in range(int(rng.integers(1, 4))):
        proto = _shape(rng, [1])
        if (proto != 0).sum() < 2:
            continue
        for col in [_c(rng, others)] + [src] * int(rng.integers(1, 3)):
            _place(rng, g, np.where(proto != 0, col, 0), gap=1)
    return g


GENERATORS.update({"blocky_noise": gen_blocky_noise, "stretched": gen_stretched, "shapepairs": gen_shapepairs})


def gen_symbbox(rng, pal, h, w, hint):
    """A partly erased symmetric pattern placed at a random offset inside a larger empty canvas."""
    mode = hint.get("mode", "h")
    k = int(rng.integers(3, 8))
    pat = GENERATORS["symhalf"](rng, pal, k, k, {"mode": mode})
    if mode not in ("rot", "d4"):
        pat = pat[:, :k]
    H, W = max(h, pat.shape[0] + 2), max(w, pat.shape[1] + 2)
    g = np.zeros((H, W), dtype=int)
    y, x = int(rng.integers(0, H - pat.shape[0] + 1)), int(rng.integers(0, W - pat.shape[1] + 1))
    g[y:y + pat.shape[0], x:x + pat.shape[1]] = pat
    return g


GENERATORS["symbbox"] = gen_symbbox


def gen_periodic(rng, pal, h, w, hint):
    py, px = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    tile = np.where(rng.random((py, px)) < .8, rng.choice(pal, size=(py, px)), BG)
    tile[0, 0] = _c(rng, pal)
    return np.tile(tile, (int(rng.integers(2, 5)), int(rng.integers(2, 5))))


def gen_parts(rng, pal, h, w, hint):
    ny, nx = int(hint.get("ny", 2)), int(hint.get("nx", 2))
    a, b = int(rng.integers(2, 7)), int(rng.integers(2, 7))
    cols = list(rng.permutation(pal))
    rows = []
    for i in range(ny):
        rows.append(np.hstack([np.where(rng.random((a, b)) < .4, cols[(i * nx + j) % len(cols)], BG)
                               for j in range(nx)]))
    return np.vstack(rows)


GENERATORS.update({"periodic": gen_periodic, "parts": gen_parts})
