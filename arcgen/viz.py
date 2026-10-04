"""Matplotlib helpers for notebooks (ARC colour palette)."""
from __future__ import annotations

import numpy as np

ARC_COLORS = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
              "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]


def draw_grid(ax, g, title=None):
    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib.colors import ListedColormap, BoundaryNorm
    g = np.asarray(g)
    ax.imshow(g, cmap=ListedColormap(ARC_COLORS), norm=BoundaryNorm(range(11), 10))
    ax.set_xticks(np.arange(-.5, g.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, g.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#444444", linewidth=.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title, fontsize=9)


def show_pairs(pairs, title=None, size=1.9):
    """pairs: list of (input, output) grids, one row of two panels per pair."""
    import matplotlib.pyplot as plt
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(2 * size * 1.4, n * size * 1.2), squeeze=False)
    for i, (a, b) in enumerate(pairs):
        draw_grid(axes[i][0], a, "input" if i == 0 else None)
        draw_grid(axes[i][1], b, "output" if i == 0 else None)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def show_task(rec: dict, max_pairs=4):
    """rec: a dataset.jsonl record (train/test pairs + program + optional story)."""
    pairs = [(np.array(p["input"]), np.array(p["output"])) for p in (rec["train"] + rec["test"])][:max_pairs]
    head = rec.get("story") or ""
    return show_pairs(pairs, f"{rec.get('family', '')} - {head}\n{rec['program_text']}")


def show_trace(grids, labels=None, size=1.7):
    """Step-by-step intermediate grids of one input (from arcgen.program.run)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(grids), figsize=(len(grids) * size * 1.3, size * 1.4), squeeze=False)
    for i, g in enumerate(grids):
        draw_grid(axes[0][i], g, (labels[i] if labels else ("input" if i == 0 else f"step {i}")))
    fig.tight_layout()
    return fig
