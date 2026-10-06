"""Layer / object / relation features of one grid. Everything is colour-label-agnostic and shape-class based.

Hierarchy:  10 colour layers  ->  objects (8-connected, one colour)  ->  cells.
Shape classes are invariant to translation, rotation, reflection (D4) and colour: a rectangle, or any complicated
shape, gets the SAME class id wherever/however it appears. Several equivalence notions are provided at once
(exact shape, shape up to D4, size, dims, holes, colour) because tasks differ in which sameness matters.
"""
from __future__ import annotations

import numpy as np

from arcgen.core import find_objects

PAD = 10                     # colour id of padding cells
REL_OO = ["same_shape", "same_shape_d4", "same_size", "same_dims", "same_dims_rot", "same_holes", "same_color",
          "touches", "contains", "inside", "row_overlap", "col_overlap", "left_of", "above"]
EQUIV = ["same_shape", "same_shape_d4", "same_size", "same_dims_rot", "same_holes", "same_color"]
REL_LL = ["same_count", "same_nobj", "contains_bbox", "bbox_overlap", "touches"]
N_OBJ_ATTR = 11              # valid, colour, size, h, w, holes, is_rect, r0, c0, size_rank_desc, size_rank_asc
MAX_MULT = 7                 # class multiplicities are clipped to 0..7
N_LAYER_ATTR = 5             # present, count, n_objects, h, w


def d4_variants(m: np.ndarray):
    out = []
    for k in range(4):
        r = np.rot90(m, k)
        out += [r, r[:, ::-1]]
    return out


def shape_keys(mask: np.ndarray):
    """(exact key, D4-canonical key) of a translation-normalised boolean mask."""
    exact = (mask.shape, mask.tobytes())
    canon = min((v.shape, np.ascontiguousarray(v).tobytes()) for v in d4_variants(mask))
    return exact, canon


def grid_features(g: np.ndarray, G: int = 30, K: int = 64) -> dict:
    g = np.asarray(g, dtype=np.int64)
    h, w = g.shape
    assert h <= G and w <= G
    color = np.full((G, G), PAD, dtype=np.int64)
    color[:h, :w] = g

    objs = find_objects(g, "c8")
    n_by_color = np.bincount([o.color for o in objs], minlength=10) if objs else np.zeros(10, int)
    objs = sorted(objs, key=lambda o: (-o.size, o.r0, o.c0))[:K]
    n = len(objs)

    obj_idx = np.full((G, G), -1, dtype=np.int64)
    attr = np.zeros((K, N_OBJ_ATTR), dtype=np.int64)
    bbox = np.zeros((K, 4), dtype=np.int64)                # r0, c0, r1, c1
    ex_keys, cn_keys = [None] * K, [None] * K
    for i, o in enumerate(objs):
        obj_idx[o.rows, o.cols] = i
        attr[i] = [1, o.color, min(o.size, 63), min(o.h, G), min(o.w, G), min(o.n_holes(), 7), int(o.is_rect),
                   o.r0, o.c0, 0, 0]
        bbox[i] = [o.r0, o.c0, o.r1, o.c1]
        ex_keys[i], cn_keys[i] = shape_keys(o.mask())

    valid = attr[:, 0] == 1
    vv = valid[:, None] & valid[None, :]
    rel = np.zeros((K, K, len(REL_OO)), dtype=bool)
    if n:
        ex = np.array([hash(k) for k in ex_keys[:n]]); cn = np.array([hash(k) for k in cn_keys[:n]])
        sz, hh, ww, ho, co = attr[:n, 2], attr[:n, 3], attr[:n, 4], attr[:n, 5], attr[:n, 1]
        eq = lambda v: v[:, None] == v[None, :]
        r0, c0, r1, c1 = bbox[:n].T
        sl = (slice(0, n), slice(0, n))
        rel[sl + (0,)] = eq(ex)
        rel[sl + (1,)] = eq(cn)
        rel[sl + (2,)] = eq(sz)
        rel[sl + (3,)] = eq(hh) & eq(ww)
        mx, mn = np.maximum(hh, ww), np.minimum(hh, ww)
        rel[sl + (4,)] = eq(mx) & eq(mn)
        rel[sl + (5,)] = eq(ho)
        rel[sl + (6,)] = eq(co)
        for dy in (-1, 0, 1):                              # touches: 8-adjacent cells of different objects
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                a = obj_idx[max(0, -dy):G - max(0, dy), max(0, -dx):G - max(0, dx)]
                b = obj_idx[max(0, dy):G - max(0, -dy) or None, max(0, dx):G - max(0, -dx) or None]
                m = (a >= 0) & (b >= 0) & (a != b)
                rel[a[m], b[m], 7] = True
        cont = ((r0[:, None] <= r0[None]) & (r1[:, None] >= r1[None]) & (c0[:, None] <= c0[None])
                & (c1[:, None] >= c1[None]) & ~(eq(r0) & eq(r1) & eq(c0) & eq(c1)))
        rel[sl + (8,)] = cont
        rel[sl + (9,)] = cont.T
        rel[sl + (10,)] = (r0[:, None] <= r1[None]) & (r0[None] <= r1[:, None])
        rel[sl + (11,)] = (c0[:, None] <= c1[None]) & (c0[None] <= c1[:, None])
        rel[sl + (12,)] = c1[:, None] < c0[None]
        rel[sl + (13,)] = r1[:, None] < r0[None]
    rel &= vv[:, :, None]
    eye = np.eye(K, dtype=bool)
    rel &= ~eye[:, :, None]

    # equivalence class ids (-1 = invalid object) for class pooling
    cls = np.full((len(EQUIV), K), -1, dtype=np.int64)
    for e, name in enumerate(EQUIV):
        r = rel[:, :, REL_OO.index(name)] | eye
        for i in range(n):
            cls[e, i] = int(np.argmax(r[i] & valid))      # smallest index of its class = canonical id

    # number features: how many objects share my class (multiplicity) and my size rank (0 = largest / smallest)
    mult = np.zeros((K, len(EQUIV)), dtype=np.int64)
    for e in range(len(EQUIV)):
        for i in range(n):
            mult[i, e] = min(int(((cls[e] == cls[e, i]) & valid).sum()), MAX_MULT)
    if n:
        sizes = np.array(sorted({int(x) for x in attr[:n, 2]}))
        attr[:n, 9] = np.minimum(len(sizes) - 1 - np.searchsorted(sizes, attr[:n, 2]), MAX_MULT)
        attr[:n, 10] = np.minimum(np.searchsorted(sizes, attr[:n, 2]), MAX_MULT)

    # layers
    layer = np.zeros((10, N_LAYER_ATTR), dtype=np.int64)
    lb = np.zeros((10, 4), dtype=np.int64)
    for c in range(10):
        ys, xs = np.nonzero(g == c)
        if len(ys):
            layer[c] = [1, min(len(ys), 63), min(int(n_by_color[c]), 31), ys.max() - ys.min() + 1, xs.max() - xs.min() + 1]
            lb[c] = [ys.min(), xs.min(), ys.max(), xs.max()]
    pres = layer[:, 0] == 1
    lrel = np.zeros((10, 10, len(REL_LL)), dtype=bool)
    pp = pres[:, None] & pres[None, :]
    lrel[:, :, 0] = (layer[:, 1][:, None] == layer[:, 1][None]) & pp
    lrel[:, :, 1] = (layer[:, 2][:, None] == layer[:, 2][None]) & pp
    a0, a1, a2, a3 = lb.T
    lrel[:, :, 2] = ((a0[:, None] <= a0[None]) & (a2[:, None] >= a2[None]) & (a1[:, None] <= a1[None])
                     & (a3[:, None] >= a3[None])) & pp
    lrel[:, :, 3] = ((a0[:, None] <= a2[None]) & (a0[None] <= a2[:, None]) & (a1[:, None] <= a3[None])
                     & (a1[None] <= a3[:, None])) & pp
    for dy, dx in ((0, 1), (1, 0)):
        a = g[:h - dy, :w - dx].ravel(); b = g[dy:, dx:].ravel()
        m = a != b
        lrel[a[m], b[m], 4] = True
        lrel[b[m], a[m], 4] = True
    lrel[np.arange(10), np.arange(10), :] = False

    return {"color": color, "obj_idx": obj_idx, "obj_attr": attr, "obj_rel": rel, "obj_cls": cls, "obj_mult": mult,
            "layer_attr": layer, "layer_rel": lrel, "shape": np.array([h, w])}


# ---- augmentation (applied to grids BEFORE feature extraction) ---------------------------------------------

def d4(g: np.ndarray, k: int) -> np.ndarray:
    r = np.rot90(g, k % 4)
    return r[:, ::-1] if k >= 4 else r


def d4_inverse(g: np.ndarray, k: int) -> np.ndarray:
    r = g[:, ::-1] if k >= 4 else g
    return np.rot90(r, -(k % 4))


def apply_perm(g: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return perm[g]


def random_perm(rng) -> np.ndarray:
    p = np.arange(10)
    p[1:] = 1 + rng.permutation(9)
    return p
