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


def test_relational_ops_known_results():
    a = lambda n, g, **p: apply_step(G(g), n, p)
    eq = lambda x, y: (x == G(y)).all()
    assert eq(a("recolor_from_marker", "1130", erase=True), "3300")
    assert eq(a("recolor_from_marker", "1130", erase=False), "3330")
    ring = "22222 20002 20102 20002 22222"
    assert eq(a("recolor_contained", ring, mode="inherit"), "22222 20002 20202 20002 22222")
    assert eq(a("recolor_contained", ring, mode="invert"), "11111 10001 10101 10001 11111")
    out = a("stamp_template", "0100000 1110000 0100000 0000000 0000000 0000030 0000000",
            marker=3, recolor=False)
    assert out[4, 5] == 1 and out[5, 4] == 1 and out[5, 5] == 1 and out[6, 5] == 1
    assert eq(a("slide_toward", "10002", mover=1, target=2), "00012")
    assert eq(a("slide_toward", "10003", mover=1, target=2), "10003")
    assert eq(a("connect_to_target", "1002", mover=1, target=2), "1112")
    assert eq(a("repair_tiling", "1212 1212 1252 1212", mask=5, crop=False), "1212 1212 1212 1212")
    assert eq(a("repair_tiling", "1212 1212 1252 1212", mask=5, crop=True), "1")
    assert eq(a("repair_symmetry", "1200", mask=0, mode="h", crop=False), "1221")
    assert eq(a("extend_periodic", "121200000", axis=1), "121212121")
    assert eq(a("fill_holes_by_area", "111 101 111", colors=[4, 5, 6]), "111 141 111")
    assert eq(a("flood_from_seed", "11111 10001 10201 10001 11111", seed=2, color=4),
              "11111 14441 14241 14441 11111")
    assert eq(a("draw_rect_between", "1000 0000 0001", fill=False, color=-1), "1111 1001 1111")
    assert eq(a("fill_largest_empty_rect", "0011 0011 1111", color=5), "5511 5511 1111")
    assert eq(a("object_histogram", "10101 00000 20200", seg="c8"), "111 220")
    assert eq(a("unify_multicolor", "112", mode="minor", seg="m8"), "222")
    assert eq(a("unify_multicolor", "112", mode="major", seg="m8"), "111")
    assert eq(a("mirror_over_line", "100 555 000", line=5), "100 555 100")
    assert eq(a("fill_cells", "10201 00200", mode="nonempty", color=-1), "11211 11211")
    assert eq(a("fill_cells", "10201 00200", mode="empty", color=4), "10201 00200")


def test_relational_selectors():
    g = G("1000 0002 0000")
    rec = lambda by, **kw: apply_step(g, "recolor_objects",
                                      {"sel": {"by": by, **kw}, "seg": "c8", "color": 7})
    assert (rec("leftmost") == G("7000 0002 0000")).all()
    assert (rec("rightmost") == G("1000 0007 0000")).all()
    assert (rec("topmost") == G("7000 0002 0000")).all()
    rt = lambda v: apply_step(G("1120 0000 0003"), "recolor_objects",
                              {"sel": {"by": "touching", "value": v}, "seg": "c8", "color": 7})
    assert (rt(2) == G("7720 0000 0003")).all()
    assert (rt(3) == G("1120 0000 0003")).all()
    d = G("1010 0000 0220")  # all single/line shapes: two 1-dots share a shape; the 22 pair is unique
    r = apply_step(d, "recolor_objects", {"sel": {"by": "dup_shape"}, "seg": "c8", "color": 7})
    assert (r == G("7070 0000 0220")).all()


def test_extra_ops_known_results():
    a = lambda n, g, **p: apply_step(G(g), n, p)
    eq = lambda x, y: (x == G(y)).all()
    assert eq(a("cells_to_pixels", "11244 10204 22222 30255 00205"), "14 35")
    assert eq(a("pool", "1100 1100 0011 0011", k=2, mode="any"), "10 01")
    assert eq(a("pool", "1101 1100", k=2, mode="all"), "10")
    assert eq(a("dedupe_adjacent", "1122 1122 3344", axis=2), "12 34")
    assert eq(a("tile_flip", "12 34", ny=1, nx=2), "1221 3443")
    assert eq(a("tile_flip", "12 34", ny=2, nx=1), "12 34 34 12")
    assert eq(a("rot_quad", "12 34", cw=False), "1224 3413 3143 4221")
    assert eq(a("shear", "123 456 789", step=1, axis=1), "123 645 897")
    assert eq(a("scale_by_colors", "12"), "1122 1122")
    assert eq(a("object_color_cell", "1100 0002", sel={"by": "largest"}, seg="c8"), "1")
    assert eq(a("swap_object_colors", "1122 0000", seg="m8"), "2211 0000")
    assert eq(a("rotate_objects", "10 11", sel={"by": "all"}, seg="c8", k=1), "01 11")
    assert eq(a("recolor_by_shape_match", "110022 000000 110000", src=1), "220022 000000 220000")


def test_search_found_ops_known_results():
    a = lambda name, g, **p: apply_step(G(g), name, p)
    eq = lambda x, y: (x == G(y)).all()
    assert eq(a("pad_replicate", "12 34", n=1, corners=True), "0120 1122 3344 0340")
    assert eq(a("pad_replicate", "12 34", n=1, corners=False), "1122 1122 3344 3344")
    assert eq(a("symmetrize_bbox", "0000 0100 0020 0000", mode="both"), "0000 0110 0220 0000")
    assert eq(a("paint_interior", "11111 11111 11111", sel={"by": "all"}, seg="c8", color=4),
              "11111 14441 11111")
    assert eq(a("recolor_split", "1022", sel={"by": "largest"}, seg="c8", yes=5, no=6), "6055")
    assert eq(a("slide_cells", "1020 0000", src=1, dir="right"), "0120 0000")
    assert eq(a("crop_to_color", "0000 0330 0300 0000", color=3, inner=False), "33 30")
    assert eq(a("full_lines", "000 010 000", src=1, dir="h"), "000 111 000")
    assert eq(a("full_lines", "000 020 000", src=2, dir="v"), "020 020 020")
    assert eq(a("connect_diag", "1000 0000 0010", src=1, line=4), "1000 0400 0010")


def test_connect_any_and_step_toward():
    a = lambda name, g, **p: apply_step(G(g), name, p)
    assert (a("connect_pairs", "10001 20003", src=-1, line=-1) == G("11111 20003")).all()
    assert (a("connect_pairs", "10002", src=-1, line=-1) == G("10002")).all()
    assert (a("step_toward", "3000 0000 0004", mover=3, target=4, steps=1) == G("0000 0300 0004")).all()


def test_shape_changing_ops():
    a = lambda name, g, **p: apply_step(G(g), name, p)
    eq = lambda x, y: (x == G(y)).all()
    assert eq(a("crop_fixed", "123 456 789", corner="br", h=2, w=2), "56 89")
    assert eq(a("crop_fixed", "123 456 789", corner="tl", h=1, w=3), "123")
    assert eq(a("extract_period", "1212 3434 1212 3434"), "12 34")
    assert eq(a("extract_period", "121212"), "12")
    assert eq(a("take_part", "1122 1122", ny=1, nx=2, idx=1), "22 22")
    assert eq(a("overlay_parts", "1102", ny=1, nx=2, order=[0, 1]), "11")
    assert eq(a("overlay_parts", "1102", ny=1, nx=2, order=[1, 0]), "12")
    assert eq(a("count_bar_fixed", "1010", sel={"by": "all"}, seg="c8", color=5, width=4), "5500")
    assert eq(a("color_histogram", "1112 0000", vertical=False), "111 200")
    assert eq(a("color_histogram", "1112 0000", vertical=True), "10 10 12")
    assert eq(a("pool", "1100 1100 0010 0000", k=2, mode="major"), "10 00")


def test_round2_ops():
    a = lambda name, g, **p: apply_step(G(g), name, p)
    eq = lambda x, y: (x == G(y)).all()
    assert eq(a("react_touching", "320 000 050", a=3, b=2, color=8), "800 000 050")
    assert eq(a("recolor_mirrored", "1201 0000", axis=1, color=7), "7207 0000")
    assert eq(a("keep_center_line", "123 456 789", axis=1), "020 050 080")
    assert eq(a("keep_center_line", "123 456 789", axis=0), "000 456 000")
    assert eq(a("crop_object_only", "110 002 000", sel={"by": "largest"}, seg="c8"), "11")
    assert eq(a("concat_with", "12 34", axis=1, tf="flip_h", swap=False), "1221 3443")
    assert eq(a("concat_with", "12 34", axis=0, tf="rot180", swap=True), "43 21 12 34")
    assert eq(a("self_logic", "1200", tf="flip_h", mode="or", color=5), "5555")
    assert eq(a("self_logic", "1200", tf="flip_h", mode="and", color=5), "0000")
    assert eq(a("recolor_by_frequency_rank", "1112 2000", colors=[5, 6]), "5556 6000")
    assert eq(a("swap_extreme_colors", "1112"), "2221")
    assert eq(a("bbox_frame_all", "000 010 000", color=4, pad=1), "444 414 444")
    assert eq(a("majority_filter", "111 121 111", conn=8), "111 111 111")
    assert eq(a("mark_centers", "111 111 111", sel={"by": "all"}, seg="c8", color=4), "111 141 111")
    assert eq(a("connect_pairs", "1000 0000 1000", src=1, line=-1, axis=0), "1000 0000 1000")
    assert eq(a("connect_pairs", "1000 0000 1000", src=1, line=-1, axis=1), "1000 1000 1000")
    assert eq(a("connect_diag", "1000 0000 0010", src=-1, line=-1), "1000 0100 0010")


def test_new_selectors():
    g = G("1010 0000 0220")
    r = lambda sel: apply_step(g, "recolor_objects", {"sel": sel, "seg": "c8", "color": 7})
    assert (r({"by": "common_shape"}) == G("7070 0000 0220")).all()
    ring = G("111 101 111 000 200")
    assert (apply_step(ring, "recolor_objects", {"sel": {"by": "holes_eq", "value": 1}, "seg": "c8",
                                                 "color": 7}) == G("777 707 777 000 200")).all()


def test_check_program_survives_garbage_params():
    from arcgen.serialize import check_program
    pair = [(G("10"), G("01"))]
    for bad in ("recolor_objects(sel=3,seg=5,color=1)", "flip(axis=[7])", "tile(ny=-3,nx=x)",
                "recolor_by_size_rank(colors=5,seg=c8)", "pad(n=4,color=true)"):
        assert check_program(bad, pair) is False


def test_curated_families_produce_verified_tasks():
    from arcgen.curated import FAMILIES, make_curated_task
    from arcgen.program import to_text
    from arcgen.serialize import check_program, parse_program
    assert len(FAMILIES) >= 30
    for name in FAMILIES:
        t = make_curated_task(3, 1, [name])
        assert t.extra["family"] == name and t.extra["story"]
        text = to_text(t.program)
        assert parse_program(text) == t.program
        assert check_program(text, t.train + t.test), name
        assert len(t.program) >= 2, name  # families are multi-step stories


def test_mixed_and_curated_generation(tmp_path):
    st = generate(30, str(tmp_path), seed=5, kind="mixed", log=lambda *_: None)
    assert st["tasks"] == 30
    recs = [json.loads(l) for l in (tmp_path / "dataset.jsonl").read_text().splitlines()]
    assert any("family" in r for r in recs) and any("family" not in r for r in recs)


def test_kaggle_branch_accepts_clear_rule_and_fails_closed():
    from arcgen.kaggle_branch import verified_arcgen_predictions
    rng = np.random.default_rng(0)
    grids = [rng.integers(0, 4, size=(5, 6)) for _ in range(4)]
    clear = {"train": [{"input": g.tolist(), "output": g[:, ::-1].tolist()} for g in grids[:3]],
             "test": [{"input": grids[3].tolist()}]}
    preds, audit = verified_arcgen_predictions(clear, time_limit=6)
    assert audit["accepted"] and (preds[0] == grids[3][:, ::-1]).all()
    noise = {"train": [{"input": g.tolist(), "output": rng.integers(0, 4, size=(5, 6)).tolist()} for g in grids[:3]],
             "test": [{"input": grids[3].tolist()}]}
    preds, audit = verified_arcgen_predictions(noise, time_limit=3)
    assert not audit["accepted"] and preds == []


def test_kaggle_notebook_builder(tmp_path):
    import ast, subprocess, sys
    src = "notebooks/original/arc-agi2-original-kg.ipynb"
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "tools/build_kaggle_notebook.py", "--src", src, "--out", str(out)], check=True,
                   capture_output=True)
    nb = json.loads(out.read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    joined = "\n".join(code)
    assert "arcgen.kaggle_branch" in joined and "with_arcgen" in joined and "hybrid_arcgen" in joined
    for c in code:  # every code cell still parses (shell/magic lines removed)
        body = "\n".join(l for l in c.splitlines() if not l.startswith(("!", "%%")))
        ast.parse(body)
    order = [i for i, c in enumerate(code) if "arcgen_predictions.jsonl" in c and "Popen" in c]
    run = [i for i, c in enumerate(code) if "python starter.py" in c]
    assert order and run and order[0] < run[0]  # background search starts before the blocking Qwen run
