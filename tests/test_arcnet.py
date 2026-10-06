import numpy as np
import torch

from arcnet.features import (EQUIV, REL_OO, apply_perm, d4, d4_inverse, grid_features, random_perm)


def G(s):
    return np.array([[int(c) for c in r] for r in s.split()])


def cls_of(f, name):
    return f["obj_cls"][EQUIV.index(name)]


def test_same_class_whatever_the_orientation_colour_or_place():
    g = np.zeros((10, 10), dtype=int)
    for (r, c) in [(0, 0), (1, 0), (2, 0), (2, 1)]: g[r, c] = 1            # L tetromino
    for (r, c) in [(0, 5), (0, 6), (0, 7), (1, 5)]: g[r, c] = 2            # rotated L, other colour, other place
    for (r, c) in [(4, 8), (5, 8), (6, 8), (6, 7)]: g[r, c] = 3            # mirrored L (J)
    for (r, c) in [(7, 2), (8, 1), (8, 2), (8, 3), (9, 2)]: g[r, c] = 4    # plus: a different shape
    f = grid_features(g, G=10, K=8)
    a = f["obj_attr"]
    ls = [i for i in range(8) if a[i, 0] and a[i, 2] == 4]
    plus = [i for i in range(8) if a[i, 0] and a[i, 2] == 5]
    assert len(ls) == 3 and len(plus) == 1
    c = cls_of(f, "same_shape_d4")
    assert len({c[i] for i in ls}) == 1                          # all three -> ONE class id
    assert c[plus[0]] not in {c[i] for i in ls}                  # the plus is another class
    assert len({cls_of(f, "same_shape")[i] for i in ls}) == 3    # exact-shape classes keep orientations apart
    assert len({cls_of(f, "same_size")[i] for i in ls}) == 1


def test_rectangles_of_any_size_orientation_dims():
    g = G("1100000 1100000 0000222 0000222 3330000 3330000 3330000")
    f = grid_features(g, G=8, K=8)
    ids = [i for i in range(8) if f["obj_attr"][i, 0]]
    assert len(ids) == 3
    rot = cls_of(f, "same_dims_rot")
    assert len({rot[i] for i in ids if f["obj_attr"][i, 2] == 6}) == 1   # 2x3 and 3x2 are the same up to rotation
    assert all(f["obj_attr"][i, 6] == 1 for i in ids)                    # all are rectangles


def test_features_equivariant_to_dihedral_and_colour_permutation():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 4, size=(6, 7))
    base = grid_features(g, G=8, K=40)
    for k in range(8):
        h = apply_perm(d4(g, k), random_perm(rng))
        f = grid_features(h, G=8, K=40)
        # the multiset of (size, D4-shape-class sizes) is unchanged: same objects, just moved/recoloured
        assert sorted(base["obj_attr"][base["obj_attr"][:, 0] == 1][:, 2]) == sorted(f["obj_attr"][f["obj_attr"][:, 0] == 1][:, 2])
        assert (base["obj_rel"][:, :, REL_OO.index("same_shape_d4")].sum() == f["obj_rel"][:, :, REL_OO.index("same_shape_d4")].sum())
    assert (d4_inverse(d4(g, 5), 5) == g).all()


def _toy_tasks(n=3, seed=0):
    rng = np.random.default_rng(seed)
    tasks = []
    for t in range(n):
        demos = []
        for _ in range(4):
            g = np.zeros((8, 8), dtype=int)
            for _ in range(4):
                y, x = rng.integers(0, 6, 2)
                g[y:y + 2, x:x + 2] = rng.integers(1, 4)
            demos.append((g, np.where(g == 1, 5, g)))
        q = np.zeros((8, 8), dtype=int); q[1:3, 1:3] = 1
        tasks.append({"id": f"toy{t}", "demos": demos, "tests": [q], "solutions": [np.where(q == 1, 5, q)]})
    return tasks


def test_ladder_variants_keep_the_rule_and_are_simpler():
    from arcnet.ladder import build_ladder
    tasks = build_ladder(_toy_tasks(), seed=1)
    n = 0
    for t in tasks:
        for lv, pairs in t["variants"].items():
            assert lv in (1, 2, 3)
            for a, b in pairs:
                n += 1
                assert ((a == 1) == (b == 5)).all()      # the toy rule (1 -> 5) is visible and unchanged
                assert a.shape == b.shape
    assert n > 10


def test_ladder_schedule_prefers_the_frontier():
    from arcnet.ladder import LadderState
    st = LadderState(1, 3)
    st.ema[0] = torch.tensor([1.0, 0.5, 0.0, 1.0]); st.refresh(0)       # level1 at the frontier
    p = st.level_probs(0, [1, 2, 3])
    assert p[0] > 3 * p[1] and p[0] > 3 * p[2]


def test_train_phase_smoke_with_ladder_and_ttrl(tmp_path):
    from arcnet.data import Sampler, evaluate
    from arcnet.ladder import LadderState, build_ladder
    from arcnet.model import LRRM
    from arcnet.train import train_phase
    tasks = build_ladder(_toy_tasks(), seed=2)
    state = LadderState(len(tasks), 3)
    idx = {t["id"]: i for i, t in enumerate(tasks)}
    model = LRRM(len(tasks), G=8, K=8, d=32, heads=2, layers=1, loops=2)
    logs = []
    train_phase(model, tasks, Sampler(tasks, 8, 8, 0, state), 6, 4, 1e-3, "cpu", state=state, probe_every=3,
                log=logs.append, warmup=2)
    train_phase(model, tasks, Sampler(tasks, 8, 8, 1, state, only=[0, 1]), 4, 4, 1e-3, "cpu", state=state,
                probe_tasks=[0, 1], probe_every=2, log=logs.append, name="ttrl", warmup=2)
    assert any("ladder probe" in l for l in logs)
    credit, n = evaluate(model, tasks, idx, "cpu")
    assert n == 3 and 0 <= credit <= 3
    assert model.A == 1


def test_augmentation_is_consistent_per_id_and_invertible():
    from arcnet.data import aug_spec
    from arcnet.features import apply_perm
    rng = np.random.default_rng(3)
    g = rng.integers(0, 5, size=(5, 7))
    assert aug_spec(4, 0)[0] == 0 and (aug_spec(4, 0)[1] == np.arange(10)).all()
    for a in range(1, 6):
        k, perm = aug_spec(4, a)
        k2, perm2 = aug_spec(4, a)
        assert k == k2 and (perm == perm2).all() and perm[0] == 0
        x = apply_perm(d4(g, k), perm)
        assert (d4_inverse(np.argsort(perm)[x], k) == g).all()
