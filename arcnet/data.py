"""ARC task loading, augmentation and batching for arcnet."""
from __future__ import annotations

import json

import numpy as np
import torch

from .features import PAD, apply_perm, d4, d4_inverse, grid_features, random_perm

BATCH_KEYS = ("color", "obj_idx", "obj_attr", "obj_rel", "obj_cls", "obj_mult", "layer_attr", "layer_rel")


def load_tasks(challenges: str, solutions: str | None = None) -> list[dict]:
    ch = json.load(open(challenges))
    sol = json.load(open(solutions)) if solutions else {}
    tasks = []
    for tid, t in ch.items():
        tasks.append({"id": tid,
                      "demos": [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]],
                      "tests": [np.array(p["input"]) for p in t["test"]],
                      "solutions": [np.array(s) for s in sol[tid]] if tid in sol else None})
    return tasks


def pad_target(out: np.ndarray, G: int) -> np.ndarray:
    t = np.full((G, G), PAD, dtype=np.int64)
    t[:out.shape[0], :out.shape[1]] = out
    return t


def aug_spec(task_idx: int, a: int):
    """Fixed augmentation #a of a task: (dihedral index, colour permutation). a = 0 is the identity.
    Each (task, a) pair has its OWN task embedding, so rules that name a colour or a direction stay consistent."""
    if a == 0:
        return 0, np.arange(10)
    rng = np.random.default_rng([task_idx, a, 12345])
    return int(rng.integers(8)), random_perm(rng)


def make_sample(inp, out, task_idx, G, K, rng, A=1, a=None):
    """One training/inference sample. ``A`` = augmentations per task; ``a`` the chosen one (random if None)."""
    a = int(rng.integers(A)) if a is None else a
    k, perm = aug_spec(task_idx, a)
    f = grid_features(apply_perm(d4(inp, k), perm), G, K)
    f = {key: f[key] for key in BATCH_KEYS}
    f["task_id"] = np.int64(task_idx * A + a)
    if out is not None:
        f["target"] = pad_target(apply_perm(d4(out, k), perm), G)
    return f


def collate(items):
    b = {}
    for key in items[0]:
        b[key] = torch.from_numpy(np.stack([np.asarray(it[key]) for it in items]))
    return b


class Sampler(torch.utils.data.IterableDataset):
    """Endless stream of augmented (input, target) samples.

    Without ``state``: demo pairs only. With a LadderState: for every task the level (0 = original demo, 1.. = simpler
    LADDER variants, see ladder.py) is drawn from the self-paced frontier weights. ``only`` restricts the tasks
    (used by the test-time TTRL phase)."""

    def __init__(self, tasks, G=30, K=64, seed=0, state=None, only=None, A=1):
        self.G, self.K, self.seed, self.state, self.A = G, K, seed, state, A
        self.pools = {}                                      # (task, level) -> [(in, out)]
        for i, t in enumerate(tasks):
            fit = lambda a, b: max(a.shape) <= G and max(b.shape) <= G
            p0 = [(a, b) for a, b in t["demos"] if fit(a, b)]
            if p0:
                self.pools[(i, 0)] = p0
            for lv, pairs in t.get("variants", {}).items():
                p = [(a, b) for a, b in pairs if fit(a, b)]
                if p and state is not None:
                    self.pools[(i, lv)] = p
        self.tasks = sorted({t for t, _ in self.pools if (only is None or t in set(only))})
        self.levels = {t: sorted(l for (tt, l) in self.pools if tt == t) for t in self.tasks}

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            t = self.tasks[int(rng.integers(len(self.tasks)))]
            lv_list = self.levels[t]
            lv = lv_list[0] if self.state is None else int(rng.choice(lv_list, p=self.state.level_probs(t, lv_list)))
            pool = self.pools[(t, lv)]
            a, b = pool[int(rng.integers(len(pool)))]
            f = make_sample(a, b, t, self.G, self.K, rng, A=self.A)
            f["level"] = np.int64(lv)
            yield f


# ---- inference with test-time augmentation ---------------------------------------------------------------------

def decode(logits: np.ndarray, G: int):
    """[G*G, 11] -> (grid or None, confidence). The grid is the top-left rectangle that is not PAD."""
    p = np.exp(logits - logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    pred = p.argmax(-1).reshape(G, G)
    conf = p.max(-1).reshape(G, G)
    non = pred != PAD
    if not non.any():
        return None, 0.0
    h = int(np.nonzero(non.any(1))[0].max()) + 1
    w = int(np.nonzero(non.any(0))[0].max()) + 1
    g = pred[:h, :w]
    if (g == PAD).any():
        g = np.where(g == PAD, 0, g)
        penalty = 0.5
    else:
        penalty = 1.0
    return g.copy(), float(conf[:h, :w].mean()) * penalty


@torch.no_grad()
def predict(model, task_idx, test_input, device="cpu", top=2, with_scores=False):
    """Vote over the task's A trained augmentations (each has its own embedding); returns up to ``top`` grids."""
    G, K, A = model.G, model.K, model.A
    items, specs = [], []
    for a in range(A):
        k, perm = aug_spec(task_idx, a)
        f = grid_features(apply_perm(d4(test_input, k), perm), G, K)
        f = {key: f[key] for key in BATCH_KEYS}
        f["task_id"] = np.int64(task_idx * A + a)
        items.append(f); specs.append((k, perm))
    b = {key: v.to(device) for key, v in collate(items).items()}
    model.eval()
    logits = model(b, return_all=False).float().cpu().numpy()
    votes = {}
    for (k, perm), lg in zip(specs, logits):
        g, conf = decode(lg, G)
        if g is None:
            continue
        g = d4_inverse(np.argsort(perm)[g], k)            # undo the colour permutation, then the dihedral transform
        v = votes.setdefault((g.shape, g.tobytes()), [0.0, g])
        v[0] += conf
    ranked = sorted(votes.values(), key=lambda v: -v[0])
    return [(v, g) for v, g in ranked[:top]] if with_scores else [g for _, g in ranked[:top]]


def evaluate(model, tasks, task_index, device="cpu"):
    """Task credit with 2 attempts (needs ``solutions``). Returns (credit, n_tasks)."""
    credit = 0.0
    for t in tasks:
        if t["solutions"] is None:
            continue
        ok = 0
        for q, sol in zip(t["tests"], t["solutions"]):
            if max(q.shape) > model.G:
                continue
            cands = predict(model, task_index[t["id"]], q, device)
            ok += any(c.shape == sol.shape and (c == sol).all() for c in cands)
        credit += ok / len(t["tests"])
    return credit, sum(1 for t in tasks if t["solutions"] is not None)


def predict_all(model, tasks, task_index, device="cpu", top=8):
    """Candidates for every test input: ({task_id: [[{'score','grid'}...] per test input]}, ARC submission dict
    with attempt_1 / attempt_2 (the input grid is the fallback when the model has no candidate))."""
    cands, sub = {}, {}
    for t in tasks:
        cands[t["id"]], sub[t["id"]] = [], []
        for q in t["tests"]:
            ranked = predict(model, task_index[t["id"]], q, device, top, True) if max(q.shape) <= model.G else []
            cands[t["id"]].append([{"score": float(v), "grid": g.tolist()} for v, g in ranked])
            grids = [g for _, g in ranked[:2]] or [q]
            sub[t["id"]].append({"attempt_1": grids[0].tolist(), "attempt_2": (grids[1] if len(grids) > 1 else grids[0]).tolist()})
    return cands, sub
