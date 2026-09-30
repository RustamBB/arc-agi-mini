"""Relational / context-dependent operations (ARC-2 style).

The answer for one object or cell depends on *other* things in the grid: a marker next
to it, a frame around it, a template elsewhere, a wall it flies towards, the periodic
pattern or symmetry that the rest of the grid implies, ...
"""
from __future__ import annotations

import numpy as np

from ..core import BG, MAX_SIZE, OpError, find_objects, label
from .base import op, pick_color, pick_present, present
from .lines import _boxes

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _two_colors(rng, g):
    p = present(g)
    a = int(rng.choice(p)) if p else pick_color(rng)
    rest = [c for c in p if c != a]
    return a, (int(rng.choice(rest)) if rest else pick_color(rng, (a,)))


# ---- markers, containers, templates ----------------------------------------------

@op("recolor_from_marker", "relation", lambda rng, g: {"erase": bool(rng.integers(0, 2))},
    gens=("markers",))
def recolor_from_marker(g, erase):
    """An object touching a single-cell marker takes the marker's colour (marker optionally removed)."""
    objs = find_objects(g, "c8")
    markers = [o for o in objs if o.size == 1]
    bodies = [o for o in objs if o.size > 1]
    out = g.copy()
    for b in bodies:
        m = np.zeros(g.shape, dtype=bool)
        m[b.rows, b.cols] = True
        cols = set()
        for mk in markers:
            r, c = int(mk.rows[0]), int(mk.cols[0])
            near = m[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            if near.any():
                cols.add((mk.color, r, c))
        if len({c for c, _, _ in cols}) == 1:
            out[b.rows, b.cols] = next(iter(cols))[0]
            if erase:
                for _, r, c in cols:
                    out[r, c] = BG
    return out


@op("recolor_contained", "relation", lambda rng, g: {"mode": str(rng.choice(["inherit", "invert"]))},
    gens=("containers",))
def recolor_contained(g, mode):
    """Objects lying inside another object's bounding box take that container's colour
    (mode 'invert': the container takes the colour of what it holds)."""
    objs = find_objects(g, "c8")
    out = g.copy()
    for b in objs:
        holders = [a for a in objs if a is not b and a.size > b.size and a.r0 < b.r0
                   and a.c0 < b.c0 and a.r1 > b.r1 and a.c1 > b.c1 and a.color != b.color]
        if holders:
            a = min(holders, key=lambda a: a.h * a.w)
            if mode == "inherit":
                out[b.rows, b.cols] = a.color
            else:
                out[a.rows, a.cols] = b.color
    return out


def _s_stamp(rng, g):
    ones = [o.color for o in find_objects(g, "c8") if o.size == 1]
    return {"marker": int(rng.choice(ones)) if ones else pick_present(rng, g),
            "recolor": bool(rng.integers(0, 2))}


@op("stamp_template", "relation", _s_stamp, gens=("template",))
def stamp_template(g, marker, recolor):
    """The (single) multi-cell object is a template; copy it centred on every marker pixel."""
    objs = find_objects(g, "m8")
    tpl = [o for o in objs if o.size > 1]
    marks = [o for o in objs if o.size == 1 and o.color == marker]
    if len(tpl) != 1 or not marks:
        raise OpError("need one template and >=1 marker")
    t = tpl[0]
    cv = t.canvas()
    cy, cx = t.h // 2, t.w // 2
    out = g.copy()
    h, w = g.shape
    for mk in marks:
        r0, c0 = int(mk.rows[0]) - cy, int(mk.cols[0]) - cx
        for i, j in zip(*np.nonzero(cv)):
            r, c = r0 + i, c0 + j
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = marker if recolor else cv[i, j]
    return out


# ---- attraction ------------------------------------------------------------------

def _ray_hit(g, cells, dy, dx):
    """Distance to and colour of the first foreign non-bg cell in front of ``cells`` (or None)."""
    h, w = g.shape
    own = set(map(tuple, cells))
    best = None
    for r, c in cells:
        k = 1
        while True:
            a, b = r + dy * k, c + dx * k
            if not (0 <= a < h and 0 <= b < w):
                break
            if (a, b) in own:
                k += 1
                continue
            if g[a, b] != BG:
                if best is None or k < best[0]:
                    best = (k, int(g[a, b]))
                break
            k += 1
    return best


def _s_attract(rng, g):
    a, b = _two_colors(rng, g)
    return {"mover": a, "target": b}


@op("slide_toward", "relation", _s_attract, gens=("attract",))
def slide_toward(g, mover, target):
    """Objects of colour `mover` fly straight towards the nearest `target` object and stop on contact."""
    objs = [o for o in find_objects(g, "c8") if o.color == mover]
    if not objs:
        raise OpError("no mover")
    out = g.copy()
    for o in objs:
        out[o.rows, o.cols] = BG
    for o in sorted(objs, key=lambda o: (o.r0, o.c0)):
        cells = list(zip(o.rows.tolist(), o.cols.tolist()))
        opts = []
        for dy, dx in D4:
            hit = _ray_hit(out, cells, dy, dx)
            if hit and hit[1] == target:
                opts.append((hit[0] - 1, dy, dx))
        if opts:
            k, dy, dx = min(opts)
            out[o.rows + dy * k, o.cols + dx * k] = o.vals
        else:
            out[o.rows, o.cols] = o.vals
    return out


@op("connect_to_target", "relation", _s_attract, gens=("attract",))
def connect_to_target(g, mover, target):
    """Every `mover` cell draws a straight line to the `target` object it faces (if the path is clear)."""
    out = g.copy()
    for r, c in zip(*np.nonzero(g == mover)):
        for dy, dx in D4:
            hit = _ray_hit(g, [(int(r), int(c))], dy, dx)
            if hit and hit[1] == target:
                for k in range(1, hit[0]):
                    out[r + dy * k, c + dx * k] = mover
    return out


# ---- pattern repair ---------------------------------------------------------------

def _mask_param(rng, g):
    return 0 if rng.random() < .5 else pick_color(rng)


def _find_period(g, known):
    h, w = g.shape
    ii, jj = np.indices(g.shape)
    best = None
    for py in range(1, h + 1):
        for px in range(1, w + 1):
            if (py, px) == (h, w) or py * px * 2 > h * w:
                continue
            if best is not None and py * px >= best[0]:
                continue
            cls = ((ii % py) * px + jj % px)
            k, v = cls[known], g[known]
            mn = np.full(py * px, 99)
            mx = np.full(py * px, -1)
            np.minimum.at(mn, k, v)
            np.maximum.at(mx, k, v)
            if (mx < 0).any() or (mn != mx).any():
                continue
            best = (py * px, mn[cls])
    return best


@op("repair_tiling", "relation",
    lambda rng, g: {"mask": _mask_param(rng, g), "crop": bool(rng.integers(0, 2))}, gens=("tiled",))
def repair_tiling(g, mask, crop):
    """Find the smallest repeating tile ignoring `mask` cells; fill the mask (or output just the patch)."""
    m = g == mask
    if not m.any() or m.all():
        raise OpError("no mask")
    best = _find_period(g, ~m)
    if best is None:
        raise OpError("no period")
    out = np.where(m, best[1], g)
    if crop:
        r, c = np.nonzero(m)
        return out[r.min():r.max() + 1, c.min():c.max() + 1].copy()
    return out


@op("repair_symmetry", "relation",
    lambda rng, g: {"mask": _mask_param(rng, g), "mode": str(rng.choice(["h", "v", "both", "rot"])),
                    "crop": bool(rng.integers(0, 2))}, gens=("symmask",))
def repair_symmetry(g, mask, mode, crop):
    """Fill `mask` cells from their mirror/rotated partners (or output only the recovered patch)."""
    m = g == mask
    if not m.any():
        raise OpError("no mask")
    if mode == "rot" and g.shape[0] != g.shape[1]:
        raise OpError("rot needs square")
    partners = {"h": [g[:, ::-1]], "v": [g[::-1]],
                "both": [g[:, ::-1], g[::-1], g[::-1, ::-1]],
                "rot": [np.rot90(g, k) for k in (1, 2, 3)]}[mode]
    out = g.copy()
    for p in partners:
        fill = (out == mask) & (p != mask)
        out[fill] = p[fill]
    if (out == mask).any():
        raise OpError("cannot repair")
    if crop:
        r, c = np.nonzero(m)
        return out[r.min():r.max() + 1, c.min():c.max() + 1].copy()
    return out


def _period_extend(row):
    nz = np.nonzero(row)[0]
    if len(nz) == 0 or nz.max() == len(row) - 1:
        return row
    n = nz.max() + 1
    for p in range(1, n // 2 + 1):
        if all(row[i] == row[i % p] for i in range(n)):
            return np.array([row[i % p] for i in range(len(row))])
    return row


@op("extend_periodic", "relation", lambda rng, g: {"axis": int(rng.integers(0, 2))}, gens=("prefixrows",))
def extend_periodic(g, axis):
    """Continue each row (axis 1) or column (axis 0) whose visible part is periodic to the border."""
    a = g if axis == 1 else g.T
    return np.array([_period_extend(r) for r in a]).astype(int).reshape(a.shape).T.copy() \
        if axis == 0 else np.array([_period_extend(r) for r in a]).astype(int)


# ---- region / geometry relations ----------------------------------------------------

def _s_area(rng, g):
    return {"colors": [pick_color(rng) for _ in range(3)]}


@op("fill_holes_by_area", "relation", _s_area, gens=("rings", "objects"))
def fill_holes_by_area(g, colors):
    """Fill enclosed empty regions with a colour that depends on their area (1 / 2-4 / 5+ cells)."""
    lab, n = label(g == BG, 4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    out = g.copy()
    for k in range(1, n + 1):
        if k in border:
            continue
        a = int((lab == k).sum())
        out[lab == k] = colors[0 if a == 1 else 1 if a <= 4 else 2]
    return out


@op("flood_from_seed", "relation",
    lambda rng, g: {"seed": pick_present(rng, g), "color": -1 if rng.random() < .3 else pick_color(rng)},
    gens=("seeded",))
def flood_from_seed(g, seed, color):
    """Paint-bucket: flood the empty region around every `seed` pixel (colour -1: the seed's own)."""
    lab, _ = label((g == BG) | (g == seed), 4)
    out = g.copy()
    for k in {int(lab[r, c]) for r, c in zip(*np.nonzero(g == seed))}:
        out[(lab == k) & (g == BG)] = seed if color < 0 else color
    return out


@op("draw_rect_between", "relation",
    lambda rng, g: {"fill": bool(rng.integers(0, 2)), "color": -1 if rng.random() < .5 else pick_color(rng)},
    gens=("corners",))
def draw_rect_between(g, fill, color):
    """Two pixels of one colour are opposite corners: draw the rectangle (outline or filled)."""
    out = g.copy()
    done = False
    for c in present(g):
        r, cc = np.nonzero(g == c)
        if len(r) != 2:
            continue
        r0, r1, c0, c1 = r.min(), r.max(), cc.min(), cc.max()
        col = c if color < 0 else color
        for i in range(r0, r1 + 1):
            for j in range(c0, c1 + 1):
                if g[i, j] == BG and (fill or i in (r0, r1) or j in (c0, c1)):
                    out[i, j] = col
        done = True
    if not done:
        raise OpError("no pixel pair")
    return out


@op("fill_largest_empty_rect", "relation", lambda rng, g: {"color": pick_color(rng)},
    gens=("sparse", "few", "noise"))
def fill_largest_empty_rect(g, color):
    """Fill the unique largest all-background rectangle."""
    h, w = g.shape
    heights = np.zeros(w, dtype=int)
    best, boxes = 0, set()
    for i in range(h):
        heights = np.where(g[i] == BG, heights + 1, 0)
        for c0 in range(w):
            mh = 10 ** 6
            for c1 in range(c0, w):
                mh = min(mh, heights[c1])
                if mh == 0:
                    break
                area = mh * (c1 - c0 + 1)
                if area > best:
                    best, boxes = area, set()
                if area == best:
                    boxes.add((i - mh + 1, i, c0, c1))
    if best < 2 or len(boxes) != 1:
        raise OpError("no unique largest rectangle")
    r0, r1, c0, c1 = next(iter(boxes))
    out = g.copy()
    out[r0:r1 + 1, c0:c1 + 1] = color
    return out


def _s_hist(rng, g):
    return {"seg": str(rng.choice(["c8", "c4"]))}


@op("object_histogram", "relation", _s_hist, gens=("objects", "shapes"))
def object_histogram(g, seg):
    """One row per colour (most objects first) whose length is that colour's object count."""
    cnt: dict = {}
    for o in find_objects(g, seg):
        cnt[o.color] = cnt.get(o.color, 0) + 1
    if len(cnt) < 2 or len(set(cnt.values())) != len(cnt):
        raise OpError("need distinct counts")
    rows = sorted(cnt.items(), key=lambda kv: -kv[1])
    n = rows[0][1]
    if n > MAX_SIZE:
        raise OpError("too wide")
    out = np.zeros((len(rows), n), dtype=int)
    for i, (c, k) in enumerate(rows):
        out[i, :k] = c
    return out


def _s_unify(rng, g):
    return {"mode": str(rng.choice(["minor", "major"])), "seg": str(rng.choice(["m8", "m4"]))}


@op("unify_multicolor", "relation", _s_unify, gens=("dotted",))
def unify_multicolor(g, mode, seg):
    """Multicolour objects become one colour: their rarest (marker dot) or commonest (denoise)."""
    out = g.copy()
    for o in find_objects(g, seg):
        vals, cnt = np.unique(o.vals, return_counts=True)
        if len(vals) < 2:
            continue
        t = cnt.min() if mode == "minor" else cnt.max()
        if (cnt == t).sum() != 1:
            raise OpError("tie")
        out[o.rows, o.cols] = vals[int(np.argmax(cnt == t))]
    return out


@op("mirror_over_line", "relation", lambda rng, g: {"line": pick_present(rng, g)}, gens=("lined",))
def mirror_over_line(g, line):
    """Reflect the content on either side of a full-width line onto the other side."""
    rows = [i for i in range(g.shape[0]) if (g[i] == line).all()]
    cols = [j for j in range(g.shape[1]) if (g[:, j] == line).all()]
    if len(rows) + len(cols) != 1:
        raise OpError("need exactly one line")
    a = g if rows else g.T
    r = (rows or cols)[0]
    out = a.copy()
    for d in range(1, max(r, a.shape[0] - r)):
        up, dn = r - d, r + d
        if up >= 0 and dn < a.shape[0]:
            out[dn] = np.where(a[dn] == BG, a[up], a[dn])
            out[up] = np.where(a[up] == BG, a[dn], a[up])
    return out if rows else out.T.copy()


@op("fill_cells", "relation",
    lambda rng, g: {"mode": str(rng.choice(["nonempty", "empty"])),
                    "color": -1 if rng.random() < .4 else pick_color(rng)}, gens=("gridded",))
def fill_cells(g, mode, color):
    """In a separator-split grid flood every non-empty (or empty) cell (colour -1: its majority colour)."""
    out = g.copy()
    for r0, r1, c0, c1 in _boxes(g):
        cell = g[r0:r1, c0:c1]
        nz = cell[cell != BG]
        if mode == "nonempty" and len(nz):
            col = int(np.bincount(nz).argmax()) if color < 0 else color
            out[r0:r1, c0:c1] = col
        elif mode == "empty" and not len(nz):
            if color < 0:
                raise OpError("needs an explicit colour")
            out[r0:r1, c0:c1] = color
    return out
