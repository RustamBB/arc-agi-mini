"""Can arcnet learn *shape equivalence*?  Synthetic relation task: objects of equal shape (up to rotation / reflection)
are recoloured to 5 (task 0) / unique-shape objects are recoloured (task 1). Shapes appear in different orientations,
positions and colours, so the model must group by shape class, not by pixels.
python tools/arcnet_sanity.py [--steps 600] [--G 10]"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arcnet.data import BATCH_KEYS, collate, make_sample, pad_target  # noqa: E402
from arcnet.features import d4, grid_features  # noqa: E402
from arcnet.model import LRRM, loss_fn  # noqa: E402

SHAPES = [np.array(s) for s in ([[1, 1], [1, 1]], [[1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [0, 1, 0]], [[1]])]


def make_pair(rng, G, task):
    for _ in range(100):
        grid = np.zeros((G, G), dtype=int)
        out = np.zeros((G, G), dtype=int)
        n = int(rng.integers(3, 6))
        kinds, placed = [], []
        pool = rng.permutation(len(SHAPES))[:3]
        for _ in range(n):
            kind = int(rng.choice(pool)); s = d4(SHAPES[kind], int(rng.integers(8)))
            for _ in range(30):
                y, x = int(rng.integers(0, G - s.shape[0] + 1)), int(rng.integers(0, G - s.shape[1] + 1))
                if (grid[max(0, y - 1):y + s.shape[0] + 1, max(0, x - 1):x + s.shape[1] + 1] == 0).all():
                    col = int(rng.integers(1, 5))
                    grid[y:y + s.shape[0], x:x + s.shape[1]][s > 0] = col
                    placed.append((kind, y, x, s, col)); break
        counts = {k: sum(1 for p in placed if p[0] == k) for k, *_ in placed}
        out = grid.copy()
        for kind, y, x, s, col in placed:
            hit = counts[kind] >= 2 if task == 0 else counts[kind] == 1
            if hit:
                out[y:y + s.shape[0], x:x + s.shape[1]][s > 0] = 5
        if (out != grid).any():
            return grid, out
    return grid, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--G", type=int, default=10)
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--task0", action="store_true", help="only task 0 (no task conditioning needed)")
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--aug", type=int, default=4)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    G, K = a.G, 10
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = LRRM(2, G=G, K=K, d=a.d, heads=4, layers=a.layers, loops=a.loops, A=a.aug)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    def batch(n):
        items = []
        for _ in range(n):
            t = 0 if a.task0 else int(rng.integers(2)); i, o = make_pair(rng, G, t)
            items.append(make_sample(i, o, t, G, K, rng, A=a.aug))
        return collate(items)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = batch(32)
        outs = model(b); loss = loss_fn(outs, b["target"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                vb = batch(128); pred = model(vb, return_all=False).argmax(-1).view(-1, G, G)
                em = (pred == vb["target"]).all((1, 2)).float().mean().item()
                changed = vb["target"] != vb["color"]                      # cells the rule must change
                acc_ch = (pred == vb["target"])[changed].float().mean().item()
                acc_all = (pred == vb["target"]).float().mean().item()
            print(f"step {step} loss {loss.item():.3f} cell-acc {acc_all:.3f} changed-cell-acc {acc_ch:.3f} "
                  f"exact-match {em:.2f} {time.time()-t0:.0f}s", flush=True)
