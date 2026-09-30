"""Colour-layer operations."""
from __future__ import annotations

import numpy as np

from ..core import BG, OpError
from .base import op, pick_color, pick_new, pick_present, present


@op("recolor", "color", lambda rng, g: {"src": pick_present(rng, g), "dst": pick_color(rng)})
def recolor(g, src, dst):
    """Replace one colour with another."""
    return np.where(g == src, dst, g)


def _s_swap(rng, g):
    a = pick_present(rng, g)
    p = [c for c in present(g) if c != a]
    return {"a": a, "b": int(rng.choice(p)) if p else pick_color(rng, (a,))}


@op("swap_colors", "color", _s_swap)
def swap_colors(g, a, b):
    """Exchange two colours."""
    return np.where(g == a, b, np.where(g == b, a, g))


@op("keep_color", "color", lambda rng, g: {"color": pick_present(rng, g)})
def keep_color(g, color):
    """Extract one colour layer; everything else becomes background."""
    return np.where(g == color, g, BG)


@op("remove_color", "color", lambda rng, g: {"color": pick_present(rng, g)})
def remove_color(g, color):
    """Delete one colour layer."""
    return np.where(g == color, BG, g)


@op("recolor_all", "color", lambda rng, g: {"color": pick_color(rng)})
def recolor_all(g, color):
    """Paint all foreground cells with a single colour."""
    return np.where(g != BG, color, BG)


@op("shift_colors", "color", lambda rng, g: {"s": int(rng.integers(1, 9))})
def shift_colors(g, s):
    """Cyclically permute colours 1..9 by s."""
    lut = np.arange(10)
    for c in range(1, 10):
        lut[c] = (c - 1 + s) % 9 + 1
    return lut[g]


@op("invert_binary", "color", lambda rng, g: {"color": pick_color(rng)})
def invert_binary(g, color):
    """Swap foreground and background."""
    return np.where(g == BG, color, BG)


@op("fill_bg", "color", lambda rng, g: {"color": pick_new(rng, g)})
def fill_bg(g, color):
    """Paint the background."""
    return np.where(g == BG, color, g)


@op("majority_fill", "color")
def majority_fill(g):
    """Fill the whole grid with its most frequent foreground colour."""
    if not present(g):
        raise OpError("empty")
    vals, cnt = np.unique(g[g != BG], return_counts=True)
    return np.full_like(g, vals[np.argmax(cnt)])


@op("dominant_color_cell", "color")
def dominant_color_cell(g):
    """Output a 1x1 grid with the most frequent foreground colour."""
    if not present(g):
        raise OpError("empty")
    vals, cnt = np.unique(g[g != BG], return_counts=True)
    if (cnt == cnt.max()).sum() > 1:
        raise OpError("tie")
    return np.array([[vals[np.argmax(cnt)]]])


@op("colors_by_frequency", "color")
def colors_by_frequency(g):
    """Column of the foreground colours sorted by decreasing frequency."""
    if not present(g):
        raise OpError("empty")
    vals, cnt = np.unique(g[g != BG], return_counts=True)
    if len(set(cnt.tolist())) != len(cnt):
        raise OpError("tie")
    return vals[np.argsort(-cnt)].reshape(-1, 1)
