# Training on RTX 4060 Ti (8 GB) + Ryzen 5 7600X (6c/12t) + 32 GB RAM

The GPU path (`--device cuda --amp`) has **not been run by me** (the sandbox has no GPU); the CPU path is tested.
All numbers below for speed / VRAM are my estimates, not measurements — do the 2-minute smoke test first.
On Windows prefer **WSL2** (or plain Linux); native Windows also works because `python -m arcgen` is now spawn-safe.

## 0. Install
```bash
git clone <repo> && cd arc-agi-mini && git checkout claude/nifty-wright-ojo5du
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build (cu124 also fine)
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

## 1. Data (CPU, ~15 min for 200k tasks; ~0.8 GB on disk, ~1.5 GB RAM when loaded)
```bash
python -m arcgen generate --n 200000 --out data/train --seed 100 --workers 12 --no-arc-files
python -m arcgen generate --n 1000   --out data/val   --seed 200 --workers 12 --no-arc-files
```

## 2. Smoke test (2 min) — find the batch size that fits
```bash
python tools/train_model.py --data data/val/dataset.jsonl --out /tmp/smoke.pt --steps 50 \
    --d 384 --layers 8 --heads 6 --maxlen 2048 --bs 8 --amp
```
The log prints `peakVRAM`. Target ≲ 6.5 GB. If it is OOM lower `--bs` (and raise `--accum` to keep `bs*accum`≈32).
Rough expectation: 15 M-param model, 2048-token examples, bs 8 ≈ 2–3 GB.

## 3. Real run (suggested: ~15 M params, ≈ 8–12 h; effective batch 32)
```bash
python tools/train_model.py --data data/train/dataset.jsonl --val data/val/dataset.jsonl \
    --out model_15m.pt --d 384 --layers 8 --heads 6 --maxlen 2048 --bs 8 --accum 4 \
    --lr 4e-4 --steps 40000 --eval-every 2000 --amp
```
That is 1.3 M examples (6.4 epochs of 200k tasks, each epoch re-samples which pairs are demos).
If VRAM allows (smoke test < 5 GB), a bigger model is the next step: `--d 512 --layers 10 --heads 8` (33 M params).
Checkpoints are written every `--eval-every` steps; `--resume model.pt` continues (weights only, LR schedule restarts).
Use `--limit 20000` to train on the first 20k tasks only when experimenting.

## 4. Evaluate
```bash
python tools/eval_model.py --model model_15m.pt --synthetic data/val/dataset.jsonl --n 300 --samples 64 --device cuda
python tools/eval_model.py --model model_15m.pt --real arc-agi_training_challenges.json --samples 128 --device cuda --out real.json
# hybrid with search (CPU processes, 1 thread each — use up to 6 shards on the 7600X)
for i in 0 1 2 3 4 5; do python tools/guided_search.py --model model_15m.pt --real arc-agi_training_challenges.json \
   --out g_$i.json --samples 64 --budget 10 --fallback 30 --shard $i/6 & done; wait
```
The reference to beat (5 M-param CPU model, 40k tasks): held-out synthetic 9.5 % train-verified,
real ARC training 2.3 %, hybrid ≈ +2–6 tasks per 125.

## Notes / recommended follow-ups
* `--maxlen 2048` lets nearly all real tasks fit (with 1024 about 19 % are skipped by the eval).
* Not implemented yet, worth adding once the baseline runs: colour-permutation / transpose augmentation of demos,
  re-weighting ops that occur in real tasks (`examples/real_task_programs.jsonl`), test-time training on the demos.
* Tell me the smoke-test speed (`step … s`) and `peakVRAM` and I will tune `--bs/--accum/--d/--layers` and the step budget.

## Diagnosing a run: `val token-acc` is misleading
Most program tokens are trivial (brackets, parameter names ≈ 97–100 % accurate), so `val token-acc` plateaus around
0.80 even when the model has learned little about *which op* to pick. Use the per-category diagnosis on a checkpoint
(copy it first — the trainer overwrites `--out` every `--eval-every` steps; it can run while training):
```bash
cp model_15m.pt ckpt.pt
python tools/diagnose_model.py --model ckpt.pt --data data/val/dataset.jsonl --n 300 --device cuda
```
Reference (5 M-param CPU model, 5000 steps): op-name accuracy 11.7 %, first-op CE 3.72 nats vs 4.35 nats for a
model that ignores the grids (first-op accuracy 12 % vs 4 % for "always the most common op"), numbers 40 %.
A healthy run shows op-name accuracy and first-op CE improving steadily; for the real test run
`tools/eval_model.py --synthetic` on the same checkpoint (execution-verified programs).
