"""Hand-designed task families: meaningful multi-step puzzles with a matching input generator.

Random op sampling rarely produces puzzles where step 2 depends on what step 1 *introduced*
("paint the biggest object red, then cut out the red thing"). A family fixes such a story:
a program template whose parameters are linked (a colour created in step 1 is the selector of
step 2), a generator that makes the story visible, and an English description.

    python -m arcgen families                      # list
    python -m arcgen generate --kind curated ...   # dataset in the usual format (+ family, story)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .program import Task, build_task

COLOR_NAMES = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "azure", "maroon"]


def cn(c: int) -> str:
    return COLOR_NAMES[c] if 0 <= c <= 9 else str(c)


def st(op: str, **params) -> dict:
    return {"op": op, "params": params}


def S(by: str, value=None) -> dict:
    return {"by": by} if value is None else {"by": by, "value": value}


class New:
    """Hands out colours that are NOT in the input palette (so outputs introduce visibly new colours)."""

    def __init__(self, rng, pal):
        self.free = [c for c in range(1, 10) if c not in pal]
        rng.shuffle(self.free)

    def __call__(self) -> int:
        return int(self.free.pop())


@dataclass
class Family:
    name: str
    gen: str
    build: Callable  # (rng, pal, new) -> (steps, story)
    npal: tuple = (2, 4)
    tags: tuple = ()


FAMILIES: dict[str, Family] = {}


def family(name, gen, npal=(2, 4), tags=()):
    def deco(fn):
        FAMILIES[name] = Family(name, gen, fn, npal, tags)
        return fn
    return deco


# ---- objects: select / recolor / extract ---------------------------------------------------

@family("highlight_extract", "objects", tags=("object", "crop"))
def _(rng, pal, new):
    n = new()
    return ([st("recolor_objects", sel=S("largest"), seg="c8", color=n),
             st("crop_object_only", sel=S("color", n), seg="c8")],
            f"Paint the largest object {cn(n)}, then cut it out on its own.")


@family("odd_shape_out", "shapes", tags=("object", "relation"))
def _(rng, pal, new):
    n = new()
    return ([st("recolor_objects", sel=S("unique_shape"), seg="c8", color=n),
             st("keep_objects", sel=S("color", n), seg="c8")],
            f"One shape has no twin. Paint it {cn(n)} and delete everything else.")


@family("big_small_split", "objects", tags=("object", "classify"))
def _(rng, pal, new):
    k, a, b = int(rng.integers(2, 6)), new(), new()
    return ([st("recolor_split", sel=S("size_gt", k), seg="c8", yes=a, no=b),
             st("keep_objects", sel=S("color", a), seg="c8")],
            f"Objects with more than {k} cells become {cn(a)}, the rest {cn(b)}; keep only the {cn(a)} ones.")


@family("rank_and_fall", "objects", tags=("object", "physics"))
def _(rng, pal, new):
    d = str(rng.choice(["up", "down", "left", "right"]))
    cols = [new() for _ in range(3)]
    return ([st("slide_objects", sel=S("all"), seg="c8", dir=d),
             st("recolor_by_size_rank", colors=cols, seg="c8")],
            f"Everything slides {d} until it hits something, then the biggest three objects are painted "
            f"{', '.join(cn(c) for c in cols)} (largest first).")


@family("layer_slide_repaint", "objects", tags=("color", "physics"))
def _(rng, pal, new):
    c, d, n = int(rng.choice(pal)), str(rng.choice(["up", "down", "left", "right"])), new()
    return ([st("keep_color", color=c), st("slide_objects", sel=S("all"), seg="c8", dir=d),
             st("recolor_all", color=n)],
            f"Keep only the {cn(c)} layer, push it {d}, and repaint it {cn(n)}.")


@family("centres_only", "objects", tags=("object", "mark"))
def _(rng, pal, new):
    n = new()
    return ([st("mark_centers", sel=S("rect"), seg="c8", color=n),
             st("keep_objects", sel=S("color", n), seg="c8")],
            f"Put a {cn(n)} dot in the middle of every solid rectangle (odd sides) and erase everything else.")


@family("isolate_crop_frame", "objects", tags=("color", "crop"))
def _(rng, pal, new):
    c, n = int(rng.choice(pal)), new()
    return ([st("keep_color", color=c), st("crop_to_content"), st("pad", n=1, color=n)],
            f"Isolate the {cn(c)} cells, crop tightly, add a {cn(n)} border ring.")


@family("denoise_then_fill", "noisyrings", tags=("object", "fill"))
def _(rng, pal, new):
    n = new()
    return ([st("delete_objects", sel=S("size_lt", 2), seg="c8"), st("fill_holes", color=n)],
            f"Erase single-pixel speckles, then flood every closed ring with {cn(n)}.")


@family("clean_dots_halo", "dotted", tags=("object", "relation"))
def _(rng, pal, new):
    n = new()
    return ([st("unify_multicolor", mode="minor", seg="m8"),
             st("outline_objects", sel=S("all"), seg="c8", color=n)],
            f"Each object takes the colour of its odd dot, then gets a {cn(n)} halo.")


@family("majority_clean_shrink", "blocky_noise", tags=("noise", "scale"))
def _(rng, pal, new):
    return ([st("majority_filter", conn=8), st("pool", k=2, mode="any")],
            "Remove speckle noise, then shrink every 2x2 block to one cell.")


# ---- relations between things ------------------------------------------------------------------

@family("marker_paint_halo", "markers", tags=("relation",))
def _(rng, pal, new):
    n = new()
    return ([st("recolor_from_marker", erase=True), st("outline_objects", sel=S("all"), seg="c8", color=n)],
            f"Shapes take the colour of the pixel touching them (the pixel disappears) and glow {cn(n)}.")


@family("container_border", "containers", tags=("relation",))
def _(rng, pal, new):
    mode, n = str(rng.choice(["inherit", "invert"])), new()
    what = ("Objects inside a frame take the frame's colour" if mode == "inherit"
            else "Frames take the colour of what they hold")
    return ([st("recolor_contained", mode=mode), st("draw_border", color=n)], f"{what}; add a {cn(n)} border.")


@family("fly_to_wall_glow", "attract", tags=("relation", "physics"))
def _(rng, pal, new):
    m, t, n = pal[0], pal[1], new()
    return ([st("slide_toward", mover=m, target=t),
             st("outline_objects", sel=S("color", m), seg="c8", color=n)],
            f"{cn(m)} objects fly straight to the {cn(t)} wall, then glow {cn(n)}.")


@family("wire_to_wall", "attract", tags=("relation", "line"))
def _(rng, pal, new):
    m, t, n = pal[0], pal[1], new()
    return ([st("connect_to_target", mover=m, target=t), st("recolor", src=m, dst=n)],
            f"Every {cn(m)} cell is wired to the {cn(t)} wall; the wires and cells turn {cn(n)}.")


@family("stamp_and_crop", "template", tags=("relation", "crop"))
def _(rng, pal, new):
    m = pal[-1]
    return ([st("stamp_template", marker=m, recolor=bool(rng.integers(0, 2))), st("crop_to_content")],
            f"Copy the pattern onto every {cn(m)} marker, then crop to everything that remains.")


@family("flood_rooms_keep", "seeded", tags=("relation", "fill"))
def _(rng, pal, new):
    s, n = pal[-1], new()
    return ([st("flood_from_seed", seed=s, color=n), st("keep_color", color=n)],
            f"Pour {cn(n)} paint into every room that has a {cn(s)} seed, then keep only the paint.")


@family("box_then_flood", "corners", tags=("relation", "fill"))
def _(rng, pal, new):
    n = new()
    return ([st("draw_rect_between", fill=False, color=-1), st("fill_holes", color=n)],
            f"Two same-coloured pixels are opposite corners: draw the box, then flood its inside with {cn(n)}.")


@family("crossing_lines", "few", npal=(2, 2), tags=("line",))
def _(rng, pal, new):
    a, b = pal[0], pal[1]
    return ([st("full_lines", src=a, dir="h"), st("full_lines", src=b, dir="v")],
            f"{cn(a)} pixels draw full horizontal lines, {cn(b)} pixels full vertical lines.")


@family("wires_with_halo", "aligned", tags=("line",))
def _(rng, pal, new):
    w, h = new(), new()
    return ([st("connect_pairs", src=-1, line=w, axis=2), st("outline_objects", sel=S("color", w), seg="c8", color=h)],
            f"Connect equal-coloured pixels sharing a row/column with {cn(w)} wires, then give wires a {cn(h)} halo.")


# ---- symmetry / periodicity -----------------------------------------------------------------------

@family("symmetry_frame", "symhalf", tags=("symmetry",))
def _(rng, pal, new):
    mode, n = str(rng.choice(["h", "v", "both"])), new()
    return ([st("symmetrize", mode=mode), st("draw_border", color=n)],
            f"Complete the pattern's mirror symmetry ({mode}) and add a {cn(n)} border.")


@family("patch_zoom", "symmask", tags=("symmetry", "scale"))
def _(rng, pal, new):
    mode = str(rng.choice(["h", "v", "both"]))
    mask = 0 if rng.random() < .5 else new()
    return ([st("repair_symmetry", mask=mask, mode=mode, crop=True), st("scale_up", k=2)],
            f"What hides under the {cn(mask)} patch? Reconstruct it from the symmetry ({mode}) and enlarge it 2x.")


@family("carpet_motif", "tiledx", tags=("periodic",))
def _(rng, pal, new):
    mask = 0 if rng.random() < .5 else new()
    return ([st("repair_tiling", mask=mask, crop=False), st("extract_period")],
            f"Repair the {cn(mask)} holes in the carpet, then output its smallest repeating motif.")


@family("motif_zoom", "periodic", tags=("periodic", "scale"))
def _(rng, pal, new):
    k = int(rng.integers(2, 4))
    return ([st("extract_period"), st("scale_up", k=k)], f"Find the repeating motif and enlarge it {k}x.")


@family("squeeze_turn", "stretched", tags=("scale", "geometry"))
def _(rng, pal, new):
    k = int(rng.integers(1, 4))
    return ([st("dedupe_adjacent", axis=2), st("rot90", k=k)], f"Collapse duplicated rows/columns, rotate {k * 90} degrees.")


@family("mirror_repaint", "small", tags=("geometry", "color"))
def _(rng, pal, new):
    mode, n = str(rng.choice(["h", "v", "both"])), new()
    return ([st("mirror_concat", mode=mode), st("recolor", src=int(rng.choice(pal)), dst=n)],
            f"Mirror-extend the picture ({mode}) and repaint one colour {cn(n)}.")


@family("flip_tiling_repaint", "small", tags=("geometry", "color"))
def _(rng, pal, new):
    ny, nx, n = int(rng.integers(1, 3)), int(rng.integers(2, 4)), new()
    return ([st("tile_flip", ny=ny, nx=nx), st("recolor", src=int(rng.choice(pal)), dst=n)],
            f"Tile {ny}x{nx} with alternating mirror images, then repaint one colour {cn(n)}.")


@family("kaleidoscope_frame", "small", tags=("symmetry", "geometry"))
def _(rng, pal, new):
    n = new()
    return ([st("rot_quad", cw=bool(rng.integers(0, 2))), st("draw_border", color=n)],
            f"Make a 2x2 rotating kaleidoscope of the square picture and frame it {cn(n)}.")


# ---- panels / grids / counting -----------------------------------------------------------------------

@family("panel_logic_zoom", "halves", tags=("logic", "scale"))
def _(rng, pal, new):
    axis, mode, n, k = int(rng.integers(0, 2)), str(rng.choice(["and", "or", "xor", "nor", "a_not_b"])), new(), int(rng.integers(2, 4))
    return ([st("half_logic", axis=axis, mode=mode, color=n), st("scale_up", k=k)],
            f"Combine the two panels with {mode.upper()} painting {cn(n)}, then enlarge {k}x.")


@family("pick_cell_zoom", "gridded", tags=("panels", "scale"))
def _(rng, pal, new):
    mode = str(rng.choice(["most", "least"]))
    return ([st("select_cell", mode=mode), st("scale_up", k=int(rng.integers(2, 4)))],
            f"Pick the {'fullest' if mode == 'most' else 'emptiest'} panel and enlarge it.")


@family("panels_to_pixels", "gridded", tags=("panels", "scale"))
def _(rng, pal, new):
    k = int(rng.integers(2, 4))
    return ([st("cells_to_pixels"), st("scale_up", k=k)], f"Summarise each panel by its main colour, then enlarge {k}x.")


@family("stack_panels_paint", "gridded", tags=("panels", "color"))
def _(rng, pal, new):
    n = new()
    return ([st("overlay_cells", order=str(rng.choice(["fwd", "rev"]))), st("recolor_all", color=n)],
            f"Stack all panels on top of each other and paint the result {cn(n)}.")


@family("count_to_bar", "objects", tags=("count",))
def _(rng, pal, new):
    c, n = int(rng.choice(pal)), new()
    return ([st("count_objects_bar", sel=S("color", c), seg="c8", color=n), st("rot90", k=1)],
            f"Count the {cn(c)} objects and draw the count as a vertical {cn(n)} bar.")


@family("sorted_bars_repaint", "bars", tags=("count", "sort"))
def _(rng, pal, new):
    n = new()
    return ([st("sort_columns_by_height", order=str(rng.choice(["asc", "desc"]))), st("recolor_all", color=n)],
            f"Sort the bars by height and repaint them {cn(n)}.")


def list_families() -> list[Family]:
    return list(FAMILIES.values())


# ---- building tasks ---------------------------------------------------------------------------------------

def make_curated_task(seed: int, index: int, names: list[str] | None = None) -> Task:
    """Deterministically build the ``index``-th curated task; sets ``task.extra`` (family, story, tags)."""
    rng = np.random.default_rng([seed, index, 7])
    pool = [FAMILIES[n] for n in names] if names else list(FAMILIES.values())
    for _ in range(300):
        fam = pool[int(rng.integers(len(pool)))]
        k = int(rng.integers(fam.npal[0], fam.npal[1] + 1))
        pal = [int(c) for c in rng.choice(np.arange(1, 10), k, replace=False)]
        steps, story = fam.build(rng, pal, New(rng, pal))
        lo = int(rng.integers(6, 11))
        hi = int(rng.integers(lo, 19))
        task = build_task(rng, steps, fam.gen, pal, lo, hi,
                          n_train=int(rng.integers(2, 5)), n_test=int(rng.integers(1, 3)))
        if task is not None:
            task.extra = {"family": fam.name, "story": story, "tags": list(fam.tags)}
            return task
    raise RuntimeError(f"could not build a curated task from {[f.name for f in pool]}")
