import argparse
import json

from .dataset import generate, grid_str, make_task, record, resolve_ops
from .ops import OPS


def _list(s):
    return [x for x in s.split(",") if x] if s else None


def main():
    ap = argparse.ArgumentParser(prog="arcgen", description="Synthetic ARC-like task generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a dataset")
    g.add_argument("--n", type=int, default=1000)
    g.add_argument("--out", default="data")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--min-len", type=int, default=1)
    g.add_argument("--max-len", type=int, default=4)
    g.add_argument("--include", help="comma separated op names to use")
    g.add_argument("--exclude", help="comma separated op names to skip")
    g.add_argument("--categories", help="geometry,color,object,line,structure")
    g.add_argument("--trace", action="store_true", help="store intermediate grids per pair")
    g.add_argument("--no-arc-files", action="store_true", help="only write dataset.jsonl")

    s = sub.add_parser("show", help="pretty-print a few tasks")
    s.add_argument("--n", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-len", type=int, default=1)
    s.add_argument("--max-len", type=int, default=3)
    s.add_argument("--include")
    s.add_argument("--categories")

    e = sub.add_parser("export", help="turn dataset.jsonl into SFT training data")
    e.add_argument("--data", default="data/dataset.jsonl")
    e.add_argument("--out", default="data/sft.jsonl")
    e.add_argument("--target", choices=["program", "output"], default="program")
    e.add_argument("--format", choices=["text", "tokens"], default="text",
                   help="text: prompt/completion strings for any LLM; tokens: ids for the closed-vocab Tokenizer")

    sub.add_parser("list-ops", help="list the operation bank")
    a = ap.parse_args()

    if a.cmd == "list-ops":
        for name, o in sorted(OPS.items(), key=lambda kv: (kv[1].category, kv[0])):
            print(f"{o.category:10s} {name:24s} {o.doc}")
        print(f"\n{len(OPS)} operations")
    elif a.cmd == "show":
        pool = resolve_ops(_list(a.include), None, _list(a.categories))
        for i in range(a.n):
            t = make_task(a.seed, i, pool, a.min_len, a.max_len)
            print("=" * 60, "\nPROGRAM:", record(t, "-", False)["program_text"])
            for k, (x, y) in enumerate(t.train[:2]):
                print(f"-- train {k} input\n{grid_str(x)}\n-- output\n{grid_str(y)}")
    elif a.cmd == "export":
        from .serialize import Tokenizer, grid_text, prompt_text
        tok = Tokenizer()
        n = 0
        with open(a.data) as f, open(a.out, "w") as o:
            for line in f:
                rec = json.loads(line)
                for k, q in enumerate(rec["test"]):  # every test pair is one example
                    if a.format == "tokens":
                        p, t = tok.encode_example(rec, k, a.target)
                        row = {"id": rec["id"], "prompt": p, "target": t}
                    else:
                        import numpy as np
                        train = [(np.array(x["input"]), np.array(x["output"])) for x in rec["train"]]
                        prompt = prompt_text(train, np.array(q["input"]))
                        comp = rec["program_text"] if a.target == "program" else grid_text(q["output"])
                        row = {"id": rec["id"], "prompt": prompt, "completion": comp}
                    o.write(json.dumps(row) + "\n")
                    n += 1
        print(f"wrote {n} examples to {a.out} (vocab size {len(tok)})")
    else:
        st = generate(a.n, a.out, a.seed, a.min_len, a.max_len, _list(a.include), _list(a.exclude),
                      _list(a.categories), a.trace, not a.no_arc_files)
        print(json.dumps(st, indent=2))


main()
