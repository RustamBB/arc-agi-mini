"""Train the program-inference GPT on synthetic data.
usage: python tools/train_model.py --data data/dataset.jsonl --out model.pt --steps 4000 --bs 8"""
import argparse, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arcgen.model import GPT, Encoder, TaskData, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="model.pt")
    ap.add_argument("--val", default=""); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--resume", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    enc = Encoder()
    data = TaskData(a.data, enc, limit=a.limit or None)
    val = TaskData(a.val, enc, seed=1) if a.val else None
    print(f"{len(data)} training tasks, vocab {len(enc.tok)}", flush=True)
    model = GPT(len(enc.tok), a.d, a.layers, a.heads)
    if a.resume:
        model.load_state_dict(torch.load(a.resume)["state"])
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    warm = min(200, a.steps // 10)
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1, step / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        ids, rows, cols, segs, mask = data.batch(a.bs)
        logits, _ = model(ids[:, :-1], rows[:, :-1], cols[:, :-1], segs[:, :-1])
        loss = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                reduction="none") * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run = loss.item() if step == 1 else .98 * run + .02 * loss.item()
        if step % 25 == 0:
            print(f"step {step} loss {run:.3f} lr {lr:.2e} {time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            save(model, a.out)
            if val:
                model.eval()
                with torch.no_grad():
                    tot = acc = n = 0
                    for _ in range(20):
                        ids, rows, cols, segs, mask = val.batch(a.bs)
                        lg, _ = model(ids[:, :-1], rows[:, :-1], cols[:, :-1], segs[:, :-1])
                        m = mask[:, 1:] > 0
                        acc += (lg.argmax(-1)[m] == ids[:, 1:][m]).sum().item(); n += m.sum().item()
                print(f"  val token-acc {acc/n:.3f}", flush=True)
                model.train()


main()
