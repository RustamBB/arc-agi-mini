"""Train arcnet from scratch on ARC tasks (demos only), with LADDER curriculum and a test-time TTRL phase.

python -m arcnet.train --train-challenges arc-agi_training_challenges.json \
    --eval-challenges arc-agi_evaluation_challenges.json --eval-solutions arc-agi_evaluation_solutions.json \
    --out arcnet.pt --steps 60000 --bs 16 --ladder --ttrl-steps 20000 --amp

Only the DEMO pairs of evaluation tasks are ever trained on (that is test-time training: allowed, it uses no test
labels); the evaluation solutions are used for scoring only.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from .data import Sampler, collate, evaluate, load_tasks
from .ladder import LEVELS, LadderState, build_ladder, n_levels, probe
from .model import LRRM, loss_fn


def make_optimizer(model, lr, wd=0.05, task_mult=10.0):
    task = [p for n, p in model.named_parameters() if n.startswith("task.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("task.")]
    return torch.optim.AdamW([{"params": rest, "lr": lr, "base": lr, "weight_decay": wd},
                              {"params": task, "lr": lr * task_mult, "base": lr * task_mult, "weight_decay": 0.0}],
                             betas=(0.9, 0.95))


def train_phase(model, tasks, sampler, steps, bs, lr, device, amp=False, workers=0, state=None, probe_tasks=None,
                probe_every=100, eval_fn=None, eval_every=0, log=print, name="train", opt=None, warmup=200):
    loader = torch.utils.data.DataLoader(sampler, batch_size=bs, num_workers=workers, collate_fn=collate,
                                         prefetch_factor=4 if workers else None)
    opt = opt or make_optimizer(model, lr)
    rng = np.random.default_rng(1)
    it = iter(loader)
    t0, run, em_run = time.time(), None, None
    model.train()
    for step in range(1, steps + 1):
        scale = min(1, step / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = g["base"] * scale
        b = {k: v.to(device) for k, v in next(it).items()}
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
            outs = model(b)
        loss = loss_fn(outs, b["target"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            em = (outs[-1].argmax(-1).view(-1, model.G, model.G) == b["target"]).all((1, 2)).float().mean().item()
        run = loss.item() if run is None else .98 * run + .02 * loss.item()
        em_run = em if em_run is None else .98 * em_run + .02 * em
        if step % 25 == 0:
            log(f"[{name}] step {step}/{steps} loss {run:.3f} exact {em_run:.3f} lr {opt.param_groups[0]['lr']:.2e} "
                f"{time.time() - t0:.0f}s")
        if state is not None and step % probe_every == 0:
            sr = probe(model, tasks if probe_tasks is None else [tasks[i] for i in probe_tasks], state, rng, device) \
                if probe_tasks is None else _probe_subset(model, tasks, probe_tasks, state, rng, device)
            model.train()
            log(f"[{name}]   ladder probe success {sr:.2f}")
        if eval_fn is not None and eval_every and (step % eval_every == 0 or step == steps):
            log(f"[{name}]   eval: {eval_fn()}")
            model.train()
    return opt


def _probe_subset(model, tasks, idxs, state, rng, device):
    from .ladder import probe as _p
    sub = [tasks[i] for i in idxs]
    class Proxy:                                   # probe() indexes the state by position in `sub`; remap to real ids
        def update(self, ti, lv, ok): state.update(idxs[ti], lv, ok)
    return _p(model, sub, Proxy(), rng, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-challenges", required=True)
    ap.add_argument("--eval-challenges", required=True)
    ap.add_argument("--eval-solutions", default="")
    ap.add_argument("--out", default="arcnet.pt")
    ap.add_argument("--G", type=int, default=30); ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--loops", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ladder", action="store_true", help="LADDER curriculum over simpler task variants")
    ap.add_argument("--ladder-mode", choices=["compositional", "fractional"], default="compositional",
                    help="compositional: add colour layers / objects one at a time or in pairs; fractional: random subsets/crops")
    ap.add_argument("--ladder-kmax", type=int, default=4, help="largest number of units kept in a compositional variant")
    ap.add_argument("--ladder-order", choices=["both", "cumulative", "pairs"], default="both",
                    help="cumulative: U1, U1+U2, U1+U2+U3...; pairs: each unit with each other unit separately; both")
    ap.add_argument("--ttrl-steps", type=int, default=0, help="test-time phase on the evaluation tasks only")
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aug", type=int, default=16, help="augmentations (dihedral x colour perm) per task, each with its own embedding")
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device

    train_tasks = load_tasks(a.train_challenges)
    eval_tasks = load_tasks(a.eval_challenges, a.eval_solutions or None)
    if a.limit_tasks:
        train_tasks, eval_tasks = train_tasks[:a.limit_tasks], eval_tasks[:a.limit_tasks]
    tasks = train_tasks + eval_tasks
    idx = {t["id"]: i for i, t in enumerate(tasks)}
    eval_idx = [idx[t["id"]] for t in eval_tasks]
    state = None
    if a.ladder:
        build_ladder(tasks, mode=a.ladder_mode, kmax=a.ladder_kmax, comp_mode=a.ladder_order)
        state = LadderState(len(tasks), n_levels(a.ladder_mode, a.ladder_kmax))
        n_var = sum(len(v) for t in tasks for v in t["variants"].values())
        print(f"ladder: {n_var} verified simpler variants over {sum(1 for t in tasks if t['variants'])} tasks", flush=True)
    model = LRRM(len(tasks), G=a.G, K=a.K, d=a.d, heads=a.heads, layers=a.layers, loops=a.loops, A=a.aug).to(device)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=device)["state"])
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {device}", flush=True)

    def eval_fn():
        credit, n = evaluate(model, eval_tasks, idx, device)
        return f"eval task credit (2 attempts, voting over the trained augmentations) {credit:.2f}/{n}" if n else "no eval solutions given"

    def save():
        torch.save({"cfg": model.cfg, "state": model.state_dict(), "ids": [t["id"] for t in tasks]}, a.out)

    opt = train_phase(model, tasks, Sampler(tasks, a.G, a.K, 0, state, A=a.aug), a.steps, a.bs, a.lr, device, a.amp, a.workers,
                      state, None, a.probe_every, eval_fn if a.eval_solutions else None, a.eval_every, name="main")
    save()
    if a.ttrl_steps:
        print("TTRL phase: evaluation tasks only (demos + their ladder variants)", flush=True)
        train_phase(model, tasks, Sampler(tasks, a.G, a.K, 1, state, only=eval_idx, A=a.aug), a.ttrl_steps, a.bs, a.lr * 0.5,
                    device, a.amp, a.workers, state, eval_idx, a.probe_every,
                    eval_fn if a.eval_solutions else None, a.eval_every, name="ttrl")
        save()
    print("final:", eval_fn() if a.eval_solutions else "done", flush=True)


if __name__ == "__main__":
    main()
