import json

import numpy as np
import pytest

from arcgen import OPS, generate, make_task, run
from arcgen.core import OpError, find_objects
from arcgen.dataset import resolve_ops
from arcgen.program import apply_step


def G(s):
    return np.array([[int(c) for c in r] for r in s.split()])


def test_bank_size_and_categories():
    assert len(OPS) >= 50
    assert {o.category for o in OPS.values()} >= {"geometry", "color", "object", "line", "structure"}


def test_objects_and_holes():
    g = G("111000 101022 111000")
    objs = find_objects(g, "c8")
    assert sorted(o.size for o in objs) == [2, 8]
    ring = [o for o in objs if o.size == 8][0]
    assert ring.n_holes() == 1 and not ring.is_rect


def test_known_results():
    a = lambda n, g, **p: apply_step(G(g), n, p)
    assert (a("fill_holes", "111 101 111", color=4) == G("111 141 111")).all()
    assert (a("fill_holes", "101 101 111", color=4) == G("101 101 111")).all()  # open ring stays
    assert (a("gravity_cells", "10 01 00", dir="down") == G("00 00 11")).all()
    halves = "10201 01201"  # A=[[1,0],[0,1]]  B=[[0,1],[0,1]]
    assert (a("half_logic", halves, axis=1, mode="or", color=5) == G("55 05")).all()
    assert (a("half_logic", halves, axis=1, mode="xor", color=5) == G("55 00")).all()
    assert (a("half_logic", halves, axis=1, mode="and", color=5) == G("00 05")).all()
    assert (a("scale_down", "1100 1100", k=2) == G("10")).all()
    assert (a("shoot_rays", "000 010 000", src=1, dir="cross") == G("010 111 010")).all()
    assert (a("connect_pairs", "10001", src=1, line=-1) == G("11111")).all()
    assert (a("slide_objects", "1000 0000 0011", sel={"by": "all"}, seg="c8", dir="right")
            == G("0001 0000 0011")).all()
    assert (a("symmetrize", "1000", mode="h") == G("1001")).all()
    assert (a("stamp_shape", "000 010 000", src=1, shape="plus", color=-1) == G("010 111 010")).all()
    assert (a("recolor_objects", "1022", sel={"by": "largest"}, seg="c8", color=7) == G("1077")).all()


def test_select_cell_and_overlay():
    g = G("10201 00201")  # cells A=[[1,0],[0,0]]  B=[[0,1],[0,1]]
    assert (apply_step(g, "select_cell", {"mode": "most"}) == G("01 01")).all()
    assert (apply_step(g, "select_cell", {"mode": "least"}) == G("10 00")).all()
    assert (apply_step(g, "overlay_cells", {"order": "fwd"}) == G("11 01")).all()


def test_errors_are_oplerrors():
    with pytest.raises(OpError):
        apply_step(G("00 00"), "crop_to_content", {})
    with pytest.raises(OpError):
        apply_step(G("12 34"), "scale_down", {"k": 2})


def test_every_op_samples_and_runs():
    rng = np.random.default_rng(0)
    from arcgen.generators import GENERATORS
    ok = set()
    for name, o in OPS.items():
        for _ in range(200):
            gen = str(rng.choice(o.gens))
            g = GENERATORS[gen](rng, [1, 2, 3], 10, 10, {})
            try:
                p = o.sample(rng, g)
                out = apply_step(g, name, json.loads(json.dumps(p, default=int)))
            except OpError:
                continue
            assert out.min() >= 0 and out.max() <= 9
            ok.add(name)
    assert ok == set(OPS), set(OPS) - ok


def test_deterministic_and_valid_tasks():
    pool = resolve_ops()
    a, b = make_task(3, 5, pool), make_task(3, 5, pool)
    assert a.program == b.program
    for i in range(40):
        t = make_task(0, i, pool)
        assert 2 <= len(t.train) <= 5 and 1 <= len(t.test) <= 2
        for x, y in t.train + t.test:  # the program really maps input -> output
            assert (run(t.program, x)[-1] == y).all()
            assert not (x.shape == y.shape and (x == y).all())


def test_generate_writes_arc_format(tmp_path):
    st = generate(20, str(tmp_path), seed=2, trace=True, log=lambda *_: None)
    assert st["tasks"] == 20
    files = list((tmp_path / "tasks").glob("*.json"))
    assert len(files) == 20
    d = json.loads(files[0].read_text())
    assert set(d) == {"train", "test"} and set(d["train"][0]) == {"input", "output"}
    rec = json.loads((tmp_path / "dataset.jsonl").read_text().splitlines()[0])
    assert rec["program"] and len(rec["traces"]) == len(rec["train"]) + len(rec["test"])


def test_program_text_roundtrip_and_tokenizer():
    from arcgen.program import to_text
    from arcgen.serialize import Tokenizer, check_program, parse_program
    tok = Tokenizer()
    pool = resolve_ops()
    for i in range(150):
        t = make_task(9, i, pool)
        text = to_text(t.program)
        assert parse_program(text) == t.program          # text -> program is lossless
        ids = tok.encode_program(text)                     # closed vocab covers every token
        assert tok.decode_program(ids) == text
        assert check_program(text, t.train + t.test)       # parsed program reproduces the task
    x, y = t.train[0]
    assert (tok.decode_grid(tok.ids(tok.grid_tokens(y))) == y).all()


def test_check_program_rejects_bad_text():
    from arcgen.serialize import check_program
    pair = [(G("10"), G("01"))]
    assert check_program("flip(axis=1)", pair)
    assert not check_program("flip(axis=0)", pair)
    assert not check_program("nonsense(x=1)", pair)
    assert not check_program("flip(axis=", pair)


def test_export(tmp_path):
    import subprocess, sys
    generate(6, str(tmp_path), seed=4, log=lambda *_: None)
    for fmt in ("text", "tokens"):
        out = tmp_path / f"{fmt}.jsonl"
        subprocess.run([sys.executable, "-m", "arcgen", "export", "--data", str(tmp_path / "dataset.jsonl"),
                        "--out", str(out), "--format", fmt], check=True, capture_output=True)
        row = json.loads(out.read_text().splitlines()[0])
        assert row["prompt"] and (row.get("completion") or row.get("target"))
