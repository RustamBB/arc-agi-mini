"""Train arcnet from scratch on ARC tasks (demos only), with LADDER curriculum and a test-time TTRL phase.

Single GPU / CPU:   python -m arcnet.train --train-challenges ... --eval-challenges ... --out arcnet.pt ...
Several GPUs (DDP): python -m torch.distributed.run --nproc_per_node=4 -m arcnet.train <same flags>

Only the DEMO pairs of evaluation tasks are ever trained on (that is test-time training: allowed, it uses no test
labels); the evaluation solutions are used for scoring only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from .data import Sampler, collate, evaluate, load_tasks, predict_all
from .ladder import LadderState, build_ladder, n_levels, probe
from .model import LRRM, loss_fn


def make_optimizer(model, lr, wd=0.05, task_mult=10.0):
    task = [p for n, p in model.named_parameters() if n.startswith("task.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("task.")]
    return torch.optim.AdamW([{"params": rest, "lr": lr, "base": lr, "weight_decay": wd},
                              {"params": task, "lr": lr * task_mult, "base": lr * task_mult, "weight_decay": 0.0}],
                             betas=(0.9, 0.95))


def train_phase(model, tasks, sampler, steps, bs, lr, device, amp=False, workers=0, state=None, probe_tasks=None,
                probe_every=100, eval_fn=None, eval_every=0, log=print, name="train", opt=None, warmup=200,
                time_budget=None, net=None, rank=0, world=1):
    """amp: False | True/'bf16' | 'fp16' (T4/P100 have no bf16). time_budget (seconds) ends the phase early and makes the
    cosine learning-rate schedule follow elapsed time. With DDP (``net`` = the DDP wrapper, ``model`` = the raw module)
    rank 0's clock decides progress / stop for everybody, so learning rates and step counts stay identical."""
    loader = torch.utils.data.DataLoader(sampler, batch_size=bs, num_workers=workers, collate_fn=collate,
                                         prefetch_factor=4 if workers else None)
    fwd = net or model
    opt = opt or make_optimizer(model, lr)
    dev_type = torch.device(device).type
    amp_dtype = None if amp in (False, None, "none") else (torch.float16 if amp == "fp16" else torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and dev_type == "cuda"))
    rng = np.random.default_rng(1 + rank)
    it = iter(loader)
    t0, run, em_run = time.time(), None, None
    model.train()
    for step in range(1, steps + 1):
        prog = step / steps if not time_budget else max(step / steps, (time.time() - t0) / time_budget)
        if world > 1:
            pt = torch.tensor([prog], device=device)
            dist.broadcast(pt, 0)
            prog = float(pt.item())
        if prog >= 1 and step > 1:
            break
        scale = min(1, step / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1))))
        for g in opt.param_groups:
            g["lr"] = g["base"] * scale
        b = {k: v.to(device) for k, v in next(it).items()}
        with torch.autocast(device_type=dev_type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            outs = fwd(b)
        loss = loss_fn(outs, b["target"])
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            em = (outs[-1].argmax(-1).view(-1, model.G, model.G) == b["target"]).all((1, 2)).float().mean().item()
        run = loss.item() if run is None else .98 * run + .02 * loss.item()
        em_run = em if em_run is None else .98 * em_run + .02 * em
        if step % 25 == 0 and rank == 0:
            log(f"[{name}] step {step} loss {run:.3f} exact {em_run:.3f} lr {opt.param_groups[0]['lr']:.2e} "
                f"{time.time() - t0:.0f}s")
        if state is not None and step % probe_every == 0:       # every rank keeps its own ladder state (no collectives)
            sr = probe(model, tasks, state, rng, device) if probe_tasks is None else \
                _probe_subset(model, tasks, probe_tasks, state, rng, device)
            model.train()
            if rank == 0:
                log(f"[{name}]   ladder probe success {sr:.2f}")
        if eval_fn is not None and eval_every and step % eval_every == 0:
            _eval(eval_fn, model, rank, world, log, name)
    if eval_fn is not None and eval_every:
        _eval(eval_fn, model, rank, world, log, name)
    return opt


def _eval(eval_fn, model, rank, world, log, name):
    if rank == 0:                                   # others wait at the barrier (long NCCL timeout is set)
        log(f"[{name}]   eval: {eval_fn()}")
    if world > 1:
        dist.barrier()
    model.train()


def _probe_subset(model, tasks, idxs, state, rng, device):
    from .ladder import probe as _p
    sub = [tasks[i] for i in idxs]

    class Proxy:                                    # probe() indexes the state by position in `sub`; remap to real ids
        def update(self, ti, lv, ok):
            state.update(idxs[ti], lv, ok)
    return _p(model, sub, Proxy(), rng, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-challenges", required=True)
    ap.add_argument("--eval-challenges", required=True)
    ap.add_argument("--eval-solutions", default="")
    ap.add_argument("--out", default="arcnet.pt")
    ap.add_argument("--outdir", default=".", help="where candidates / submission files are written")
    ap.add_argument("--G", type=int, default=30); ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--loops", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=16,
                    help="per-GPU batch size")
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
    ap.add_argument("--workers", type=int, default=8, help="data-loader workers per process")
    ap.add_argument("--aug", type=int, default=16, help="augmentations (dihedral x colour perm) per task, each with its own embedding")
    ap.add_argument("--amp", nargs="?", const="bf16", default=False, choices=["bf16", "fp16"], help="mixed precision (fp16 for T4/P100)")
    ap.add_argument("--minutes", type=float, default=0, help="wall-clock budget: 70%% main phase, 30%% TTRL phase")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--top", type=int, default=8, help="candidates per test input to export")
    a = ap.parse_args()

    rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available() and a.device in ("auto", "cuda")
    device = f"cuda:{local}" if cuda and world > 1 else ("cuda" if cuda else ("cpu" if a.device == "auto" else a.device))
    if world > 1:
        if cuda:
            torch.cuda.set_device(local)
        dist.init_process_group("nccl" if cuda else "gloo", timeout=datetime.timedelta(minutes=120))
    say = (lambda *m, **k: print(*m, flush=True, **k)) if rank == 0 else (lambda *m, **k: None)
    torch.manual_seed(0)                              # identical initial weights on every rank

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
        say(f"ladder: {n_var} verified simpler variants over {sum(1 for t in tasks if t['variants'])} tasks")
    model = LRRM(len(tasks), G=a.G, K=a.K, d=a.dim, heads=a.heads, layers=a.layers, loops=a.loops, A=a.aug).to(device)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=device)["state"])
    net = None
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if cuda else None)
    say(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | {world} process(es) on {device} "
        f"| global batch {a.bs * world}")

    def eval_fn():
        credit, n = evaluate(model, eval_tasks, idx, device)
        return f"eval task credit (2 attempts, voting over the trained augmentations) {credit:.2f}/{n}" if n else \
            "no eval solutions given"

    def save():
        if rank == 0:
            torch.save({"cfg": model.cfg, "state": model.state_dict(), "ids": [t["id"] for t in tasks]}, a.out)

    ef = eval_fn if a.eval_solutions else None
    main_budget = a.minutes * 60 * (0.7 if a.ttrl_steps else 1.0) if a.minutes else None
    opt = train_phase(model, tasks, Sampler(tasks, a.G, a.K, 1000 * rank, state, A=a.aug), a.steps, a.bs, a.lr, device,
                      a.amp, a.workers, state, None, a.probe_every, ef, a.eval_every, say, "main",
                      time_budget=main_budget, net=net, rank=rank, world=world)
    save()
    if a.ttrl_steps:
        say("TTRL phase: evaluation tasks only (demos + their ladder variants)")
        train_phase(model, tasks, Sampler(tasks, a.G, a.K, 1000 * rank + 1, state, only=eval_idx, A=a.aug), a.ttrl_steps,
                    a.bs, a.lr * 0.5, device, a.amp, a.workers, state, eval_idx, a.probe_every, ef, a.eval_every, say,
                    "ttrl", time_budget=a.minutes * 60 * 0.3 if a.minutes else None, net=net, rank=rank, world=world)
        save()
    if rank == 0:
        say("final:", eval_fn() if a.eval_solutions else "done")
        cands, sub = predict_all(model, eval_tasks, idx, device, top=a.top)
        os.makedirs(a.outdir, exist_ok=True)
        json.dump(cands, open(os.path.join(a.outdir, "arcnet_candidates.json"), "w"))
        json.dump(sub, open(os.path.join(a.outdir, "submission_arcnet.json"), "w"))
        say(f"wrote {a.out}, arcnet_candidates.json, submission_arcnet.json in {a.outdir}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
