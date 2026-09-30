"""Sample programs from a trained model and verify them by execution.

python tools/eval_model.py --model model.pt --synthetic val.jsonl --n 200            # held-out synthetic tasks
python tools/eval_model.py --model model.pt --real challenges.json --out real.json   # real ARC tasks (train-verified)
Options: --samples N (per task), --temp, --shard i/k (run k processes with 1 thread each), --threads
A program counts as *train-verified* when it reproduces every demonstration pair; for synthetic tasks we
also report whether it reproduces the held-out test pair (real tasks ship without test outputs).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arcgen.model import Encoder, load  # noqa: E402
from arcgen.serialize import check_program  # noqa: E402


def predict(model, enc, train, query, n, temp):
    maxlen = model.cfg["maxlen"] - 128
    """Returns (list of distinct program texts, n_demos used) or (None, 0) if the prompt cannot fit."""
    demos = list(train)
    while demos:
        p = enc.prompt(demos, query)
        if len(p[0]) <= maxlen - 100:
            break
        demos.pop()
    if not demos:
        return None, 0
    outs = model.sample(enc, p, n=1, greedy=True) + (model.sample(enc, p, n=n, temp=temp) if n else [])
    texts = []
    for o in outs:
        t = enc.tok.decode_program(o)
        if t and t not in texts:
            texts.append(t)
    return texts, len(demos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--synthetic"); ap.add_argument("--real"); ap.add_argument("--out", default="")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, enc = load(a.model, a.device), Encoder()
    si, sk = map(int, a.shard.split("/"))
    res, t0 = {}, time.time()
    if a.synthetic:
        recs = [json.loads(l) for _, l in zip(range(a.n), open(a.synthetic))]
        for i, r in enumerate(recs):
            if i % sk != si:
                continue
            tr = [(np.array(p["input"]), np.array(p["output"])) for p in r["train"]]
            te = [(np.array(p["input"]), np.array(p["output"])) for p in r["test"]]
            texts, nd = predict(model, enc, tr, te[0][0], a.samples, a.temp)
            if texts is None:
                res[r["id"]] = {"skipped": True}
                continue
            ver = [t for t in texts if check_program(t, tr)]
            res[r["id"]] = {"greedy_exact": texts[0] == r["program_text"], "greedy_train": check_program(texts[0], tr),
                            "train_verified": bool(ver), "generalises": any(check_program(t, tr + te) for t in ver),
                            "valid": sum(check_program(t, tr) or _parses(t) for t in texts) / len(texts), "n_ops": len(r["ops"])}
    if a.real:
        d = json.load(open(a.real))
        for i, (k, v) in enumerate(d.items()):
            if i % sk != si:
                continue
            tr = [(np.array(p["input"]), np.array(p["output"])) for p in v["train"]]
            texts, nd = predict(model, enc, tr, np.array(v["test"][0]["input"]), a.samples, a.temp)
            if texts is None:
                res[k] = {"skipped": True}
                continue
            ver = [t for t in texts if check_program(t, tr)]
            res[k] = {"train_verified": bool(ver), "programs": ver[:3], "n_demos": nd}
    if a.out:
        json.dump(res, open(a.out, "w"))
    done = [v for v in res.values() if not v.get("skipped")]
    print(f"shard {a.shard}: {len(done)} evaluated, {len(res)-len(done)} skipped (too long), {time.time()-t0:.0f}s")
    for key in ("greedy_exact", "greedy_train", "train_verified", "generalises", "valid"):
        if done and key in done[0]:
            print(f"  {key}: {np.mean([v[key] for v in done]):.3f}")


def _parses(t):
    from arcgen.serialize import parse_program
    try:
        parse_program(t); return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
