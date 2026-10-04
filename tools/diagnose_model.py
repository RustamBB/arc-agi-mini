"""Does the model actually use the grids? Teacher-forced diagnosis per token category.

python tools/diagnose_model.py --model model.pt --data data/val/dataset.jsonl [--n 300] [--device cuda]

Reports, for the program tokens of held-out tasks, accuracy and cross-entropy (nats) per category, and compares the
cross-entropy of the FIRST op token with the entropy of the op distribution in the data (what a model that ignores
the grids would get). If model CE ~ prior entropy, the model has only learned the format/priors, not the task inference.
"""
import argparse, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arcgen.model import Encoder, TaskData, load  # noqa: E402
from arcgen.ops import OPS  # noqa: E402


def category(tok, prev):
    if tok in OPS:
        return "op name"
    if tok in "(){}[],=|":
        return "punctuation"
    if tok == "<eos>":
        return "eos"
    if tok.lstrip("-").isdigit():
        return "number (colour/size/offset)"
    if prev == "=":
        return "enum value"
    return "param/enum name"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model, enc = load(a.model, a.device), Encoder()
    maxlen = model.cfg["maxlen"] - 128
    data = TaskData(a.data, enc, limit=a.n, seed=1, maxlen=maxlen)
    itos = enc.tok.itos
    first_ops = Counter()
    for pairs, text in data.tasks:
        first_ops[text.split("(")[0]] += 1
    tot = sum(first_ops.values())
    prior_h = -sum(c / tot * math.log(c / tot) for c in first_ops.values())
    ce, acc, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    first_ce = first_acc = 0.0
    with torch.no_grad():
        for _ in range(a.n):
            arrs, n_prompt = data.sample()
            t = [torch.from_numpy(x.astype("int64"))[None].to(a.device) for x in arrs]
            lg, _ = model(t[0][:, :-1], t[1][:, :-1], t[2][:, :-1], t[3][:, :-1])
            lp = F.log_softmax(lg[0].float(), -1)
            ids = t[0][0]
            for pos in range(n_prompt, len(ids)):  # predict ids[pos] from position pos-1
                tok, prev = itos[int(ids[pos])], itos[int(ids[pos - 1])]
                c = category(tok, prev)
                nll = -lp[pos - 1, ids[pos]].item()
                ok = int(lp[pos - 1].argmax()) == int(ids[pos])
                ce[c] += nll; acc[c] += ok; cnt[c] += 1
                if pos == n_prompt:
                    first_ce += nll; first_acc += ok
    print(f"{'category':32s}{'tokens':>8s}{'acc':>8s}{'CE nats':>9s}")
    for c in sorted(cnt, key=lambda k: -cnt[k]):
        print(f"{c:32s}{cnt[c]:8d}{acc[c]/cnt[c]:8.3f}{ce[c]/cnt[c]:9.3f}")
    n = a.n
    print(f"\nfirst op token: model CE {first_ce/n:.3f} nats, accuracy {first_acc/n:.3f}")
    print(f"                prior entropy of first op (ignores grids) {prior_h:.3f} nats, "
          f"most-common-op accuracy {max(first_ops.values())/tot:.3f}")
    print("-> model CE well below the prior entropy means the grids are being used; ~equal means they are not.")


if __name__ == "__main__":
    main()
