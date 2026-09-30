"""Model-guided beam search on a real ARC set.
python tools/guided_search.py --model model.pt --real challenges.json --out g.json [--stride 8] [--samples 48] [--budget 20] --shard i/k
Each task: sample programs from the model -> (1) verified sample, (2) op skeletons + parameter search,
(3) search restricted to the model's ops. Solved = reproduces every train pair."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arcgen.guided import guided_solve  # noqa: E402
from arcgen.model import Encoder, load  # noqa: E402
from arcgen.program import to_text  # noqa: E402
from arcgen.search import search  # noqa: E402
from arcgen.serialize import check_program  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_model import predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--real", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--budget", type=float, default=20); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fallback", type=float, default=0, help="seconds of plain beam search if guidance fails")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--temp", type=float, default=0.9)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model, enc = load(a.model), Encoder()
    si, sk = map(int, a.shard.split("/"))
    d = json.load(open(a.real))
    res, t0 = {}, time.time()
    for i, (k, v) in enumerate(d.items()):
        if i % a.stride or (i // a.stride) % sk != si:
            continue
        pairs = [(np.array(p["input"]), np.array(p["output"])) for p in v["train"]]
        texts, _ = predict(model, enc, pairs, np.array(v["test"][0]["input"]), a.samples, a.temp)
        if texts is None:
            texts = []  # prompt too long: no model guidance (falls through to nothing)
        prog, how = guided_solve(pairs, texts, budget=a.budget, seed=i)
        if prog is None and a.fallback:
            prog, _, _ = search(pairs, depth=3, beam=4, tries=60, top=24, budget=a.fallback, seed=i)
            if prog and check_program(to_text(prog), pairs):
                how = "beam"
            else:
                prog = None
        res[k] = {"program": prog, "how": how, "guided": bool(texts)}
    json.dump(res, open(a.out, "w"))
    n = sum(1 for r in res.values() if r["program"])
    print(f"shard {a.shard}: solved {n}/{len(res)} in {time.time()-t0:.0f}s", flush=True)


main()
