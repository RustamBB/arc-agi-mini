"""Whole-grid geometric / structural operations."""
from __future__ import annotations

import numpy as np

from ..core import BG, MAX_SIZE, OpError
from .base import op, pick_color, pick_present


def _check(g):
    if g.shape[0] > MAX_SIZE or g.shape[1] > MAX_SIZE:
        raise OpError("too large")
    return g


def _overlay(a, b):
    return np.where(a != BG, a, b)


@op("rot90", "geometry", lambda rng, g: {"k": int(rng.integers(1, 4))})
def rot90(g, k):
    """Rotate counter-clockwise by k*90 degrees."""
    return np.rot90(g, k).copy()


@op("flip", "geometry", lambda rng, g: {"axis": int(rng.integers(0, 2))})
def flip(g, axis):
    """Flip up-down (axis 0) or left-right (axis 1)."""
    return np.flip(g, axis).copy()


@op("transpose", "geometry")
def transpose(g):
    """Main-diagonal reflection."""
    return g.T.copy()


@op("anti_transpose", "geometry")
def anti_transpose(g):
    """Anti-diagonal reflection."""
    return g[::-1, ::-1].T.copy()


def _s_translate(rng, g):
    while True:
        dy, dx = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
        if dy or dx:
            return {"dy": dy, "dx": dx, "wrap": bool(rng.integers(0, 2))}


@op("translate", "geometry", _s_translate)
def translate(g, dy, dx, wrap):
    """Shift the whole grid; wrap around or drop what falls off."""
    h, w = g.shape
    if wrap:
        return np.roll(g, (dy, dx), (0, 1))
    if abs(dy) >= h or abs(dx) >= w:
        raise OpError("shift too large")
    out = np.full_like(g, BG)
    out[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
        g[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return out


@op("crop_to_content", "geometry")
def crop_to_content(g):
    """Crop to the bounding box of all non-background cells."""
    r, c = np.nonzero(g != BG)
    if len(r) == 0:
        raise OpError("empty")
    return g[r.min():r.max() + 1, c.min():c.max() + 1].copy()


@op("pad", "geometry", lambda rng, g: {"n": int(rng.integers(1, 3)), "color": pick_color(rng)})
def pad(g, n, color):
    """Surround the grid with n rows/cols of a colour."""
    return _check(np.pad(g, n, constant_values=color))


@op("scale_up", "geometry", lambda rng, g: {"k": int(rng.integers(2, 4))})
def scale_up(g, k):
    """Enlarge every cell into a k x k block."""
    if g.shape[0] * k > MAX_SIZE or g.shape[1] * k > MAX_SIZE:
        raise OpError("too large")
    return np.kron(g, np.ones((k, k), dtype=int))


@op("scale_down", "geometry", lambda rng, g: {"k": int(rng.integers(2, 4))}, gens=("blocky",))
def scale_down(g, k):
    """Inverse of scale_up; blocks must be uniform."""
    h, w = g.shape
    if h % k or w % k:
        raise OpError("not divisible")
    b = g.reshape(h // k, k, w // k, k)
    if not (b == b[:, :1, :, :1]).all():
        raise OpError("blocks not uniform")
    return b[:, 0, :, 0].copy()


@op("tile", "geometry",
    lambda rng, g: {"ny": int(rng.integers(1, 4)), "nx": int(rng.integers(1, 4))},
    gens=("small", "few", "sparse"))
def tile(g, ny, nx):
    """Repeat the grid ny x nx times."""
    if ny * nx < 2:
        raise OpError("noop")
    return _check(np.tile(g, (ny, nx)))


@op("mirror_concat", "geometry",
    lambda rng, g: {"mode": str(rng.choice(["h", "v", "both"]))}, gens=("small", "few", "objects"))
def mirror_concat(g, mode):
    """Append mirrored copies (right / below / 2x2 kaleidoscope)."""
    if mode == "h":
        out = np.hstack([g, g[:, ::-1]])
    elif mode == "v":
        out = np.vstack([g, g[::-1]])
    else:
        top = np.hstack([g, g[:, ::-1]])
        out = np.vstack([top, top[::-1]])
    return _check(out)


@op("symmetrize", "geometry",
    lambda rng, g: {"mode": str(rng.choice(["h", "v", "both", "rot"]))}, gens=("symhalf",))
def symmetrize(g, mode):
    """Complete a symmetric pattern by overlaying mirrored/rotated copies."""
    if mode == "h":
        return _overlay(g, g[:, ::-1])
    if mode == "v":
        return _overlay(g, g[::-1])
    if mode == "both":
        a = _overlay(g, g[:, ::-1])
        return _overlay(a, a[::-1])
    if g.shape[0] != g.shape[1]:
        raise OpError("rot needs square")
    out = g
    for k in (1, 2, 3):
        out = _overlay(out, np.rot90(g, k))
    return out


@op("half", "geometry",
    lambda rng, g: {"which": str(rng.choice(["top", "bottom", "left", "right"]))})
def half(g, which):
    """Keep one half of the grid (the middle line of odd sizes is dropped)."""
    h, w = g.shape
    if which in ("top", "bottom") and h < 2 or which in ("left", "right") and w < 2:
        raise OpError("too small")
    return {"top": lambda: g[:h // 2], "bottom": lambda: g[(h + 1) // 2:],
            "left": lambda: g[:, :w // 2], "right": lambda: g[:, (w + 1) // 2:]}[which]().copy()


def _s_half_logic(rng, g):
    return {"axis": int(rng.integers(0, 2)),
            "mode": str(rng.choice(["and", "or", "xor", "nor", "a_not_b"])),
            "color": pick_color(rng)}


@op("half_logic", "geometry", _s_half_logic, gens=("halves",))
def half_logic(g, axis, mode, color):
    """Split into two halves (optionally separated by a line) and combine as boolean masks."""
    n = g.shape[axis]
    if n < 3:
        raise OpError("too small")
    a, b = (g[:, :n // 2], g[:, (n + 1) // 2:]) if axis == 1 else (g[:n // 2], g[(n + 1) // 2:])
    a, b = a != BG, b != BG
    m = {"and": a & b, "or": a | b, "xor": a ^ b, "nor": ~(a | b), "a_not_b": a & ~b}[mode]
    return np.where(m, color, BG)


@op("fractal", "geometry", gens=("small",))
def fractal(g):
    """Replace every non-background cell with a copy of the grid (self-similar)."""
    h, w = g.shape
    if h * h > MAX_SIZE or w * w > MAX_SIZE:
        raise OpError("too large")
    return np.kron((g != BG).astype(int), g)


@op("draw_border", "geometry", lambda rng, g: {"color": pick_color(rng)})
def draw_border(g, color):
    """Paint the outermost ring of the grid."""
    out = g.copy()
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = color
    return out


@op("remove_border", "geometry")
def remove_border(g):
    """Crop away the outer ring."""
    if min(g.shape) < 3:
        raise OpError("too small")
    return g[1:-1, 1:-1].copy()


@op("draw_grid_lines", "geometry",
    lambda rng, g: {"period": int(rng.integers(2, 5)), "color": pick_color(rng)})
def draw_grid_lines(g, period, color):
    """Paint every period-th row and column."""
    out = g.copy()
    out[period - 1::period, :] = color
    out[:, period - 1::period] = color
    return out
