"""Grid primitives: connected-component labelling, objects, selectors.

A grid is a 2-D ``np.ndarray`` of ints in 0..9, colour 0 is the background.
Objects are the "layers" the operation bank works on: a grid is decomposed
into objects (per-colour or multicolour components, 4- or 8-connected),
an operation edits a selection of them, and the result is recomposed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BG = 0
MAX_SIZE = 30


class OpError(Exception):
    """Raised when an operation is not applicable to a grid."""


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def label(mask: np.ndarray, conn: int = 8) -> tuple[np.ndarray, int]:
    """Connected-component labelling of a boolean mask (labels start at 1)."""
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=int)
    nb = N8 if conn == 8 else N4
    n = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                lab[i, j] = n
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    for dy, dx in nb:
                        a, b = y + dy, x + dx
                        if 0 <= a < h and 0 <= b < w and mask[a, b] and lab[a, b] == 0:
                            lab[a, b] = n
                            stack.append((a, b))
    return lab, n


@dataclass
class Obj:
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def r0(self) -> int:
        return int(self.rows.min())

    @property
    def r1(self) -> int:
        return int(self.rows.max())

    @property
    def c0(self) -> int:
        return int(self.cols.min())

    @property
    def c1(self) -> int:
        return int(self.cols.max())

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def color(self) -> int:
        vals, counts = np.unique(self.vals, return_counts=True)
        return int(vals[np.argmax(counts)])

    def mask(self) -> np.ndarray:
        m = np.zeros((self.h, self.w), dtype=bool)
        m[self.rows - self.r0, self.cols - self.c0] = True
        return m

    def canvas(self) -> np.ndarray:
        a = np.zeros((self.h, self.w), dtype=int)
        a[self.rows - self.r0, self.cols - self.c0] = self.vals
        return a

    def touches_border(self, shape) -> bool:
        return (self.r0 == 0 or self.c0 == 0
                or self.r1 == shape[0] - 1 or self.c1 == shape[1] - 1)

    @property
    def is_rect(self) -> bool:
        return self.size == self.h * self.w

    def n_holes(self) -> int:
        m = np.pad(self.mask(), 1)
        lab, n = label(~m, 4)
        outside = lab[0, 0]
        return len({int(v) for v in np.unique(lab) if v not in (0, outside)})


_OBJ_CACHE: dict = {}


def find_objects(g: np.ndarray, seg: str = "c8") -> list[Obj]:
    """Cached wrapper: objects are treated as read-only by every op."""
    key = (g.tobytes(), g.shape, seg)
    hit = _OBJ_CACHE.get(key)
    if hit is None:
        if len(_OBJ_CACHE) > 4000:
            _OBJ_CACHE.clear()
        hit = _OBJ_CACHE[key] = _find_objects(g, seg)
    return list(hit)


def _find_objects(g: np.ndarray, seg: str = "c8") -> list[Obj]:
    """Split ``g`` into objects. ``seg`` = ``c|m`` + ``4|8``:
    ``c`` separates by colour, ``m`` allows multicolour objects."""
    conn = 8 if seg.endswith("8") else 4
    multi = seg.startswith("m")
    groups = []
    if multi:
        groups.append(g != BG)
    else:
        groups.extend(g == c for c in np.unique(g) if c != BG)
    objs: list[Obj] = []
    for m in groups:
        lab, n = label(m, conn)
        for k in range(1, n + 1):
            r, c = np.nonzero(lab == k)
            objs.append(Obj(r, c, g[r, c]))
    objs.sort(key=lambda o: (o.r0, o.c0, o.size))
    return objs


def shape_key(o: Obj):
    return (o.mask().tobytes(), o.h, o.w)


def _touching(o: Obj, g, v) -> bool:
    m = np.zeros(g.shape, dtype=bool)
    m[o.rows, o.cols] = True
    ring = dilate_mask(m, 8) & ~m
    return bool((g[ring] == v).any())


def select(objs: list[Obj], sel: dict, shape, g=None) -> list[Obj]:
    """Choose objects with a declarative selector, e.g. ``{"by": "largest"}``.

    Relational selectors (``touching``) need the grid ``g``."""
    if not objs:
        return []
    by = sel["by"]
    v = sel.get("value")
    if by == "all":
        return list(objs)
    if by == "largest":
        m = max(o.size for o in objs)
        return [o for o in objs if o.size == m]
    if by == "smallest":
        m = min(o.size for o in objs)
        return [o for o in objs if o.size == m]
    if by == "tallest":
        m = max(o.h for o in objs)
        return [o for o in objs if o.h == m]
    if by == "widest":
        m = max(o.w for o in objs)
        return [o for o in objs if o.w == m]
    ext = {"leftmost": lambda o: o.c0, "topmost": lambda o: o.r0,
           "rightmost": lambda o: -o.c1, "bottommost": lambda o: -o.r1}
    if by in ext:
        m = min(ext[by](o) for o in objs)
        return [o for o in objs if ext[by](o) == m]
    if by in ("common_color", "rare_color"):
        cnt: dict = {}
        for o in objs:
            cnt[o.color] = cnt.get(o.color, 0) + 1
        t = max(cnt.values()) if by == "common_color" else min(cnt.values())
        return [o for o in objs if cnt[o.color] == t]
    if by in ("dup_shape", "unique_shape"):
        cnt = {}
        for o in objs:
            cnt[shape_key(o)] = cnt.get(shape_key(o), 0) + 1
        return [o for o in objs if (cnt[shape_key(o)] > 1) == (by == "dup_shape")]
    tests = {
        "touching": lambda o: g is not None and _touching(o, g, v),
        "square": lambda o: o.is_rect and o.h == o.w,
        "symmetric": lambda o: bool((o.mask() == o.mask()[:, ::-1]).all()
                                    or (o.mask() == o.mask()[::-1]).all()),
        "color": lambda o: o.color == v,
        "not_color": lambda o: o.color != v,
        "size_eq": lambda o: o.size == v,
        "size_gt": lambda o: o.size > v,
        "size_lt": lambda o: o.size < v,
        "touches_border": lambda o: o.touches_border(shape),
        "inner": lambda o: not o.touches_border(shape),
        "rect": lambda o: o.is_rect,
        "not_rect": lambda o: not o.is_rect,
        "has_hole": lambda o: o.n_holes() > 0,
        "no_hole": lambda o: o.n_holes() == 0,
        "multicolor": lambda o: len(np.unique(o.vals)) > 1,
    }
    if by not in tests:
        raise OpError(f"unknown selector {by}")
    return [o for o in objs if tests[by](o)]


def shift_mask(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return mask moved by (dy, dx); cells shifted out are dropped."""
    h, w = m.shape
    out = np.zeros_like(m)
    out[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
        m[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return out


def dilate_mask(m: np.ndarray, conn: int = 8) -> np.ndarray:
    out = m.copy()
    for dy, dx in (N8 if conn == 8 else N4):
        out |= shift_mask(m, dy, dx)
    return out
