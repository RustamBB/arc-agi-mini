# Training the program-inference model on a local GPU

Everything below runs the same code as the CPU experiments (`tools/train_model.py` now has `--device/--amp/--maxlen`;
**the GPU path itself has not been run yet** — it only differs by `.to(device)` and bf16 autocast).

```bash
git clone <repo> && cd arc-agi-mini && git checkout claude/nifty-wright-ojo5du
pip install numpy torch

# 1. data: 200k tasks (~150 tasks/s per core; use all cores) + a fixed validation set with a different seed
python -m arcgen generate --n 200000 --out data/train --seed 100 --workers 16 --no-arc-files
python -m arcgen generate --n 1000   --out data/val   --seed 200 --workers 16 --no-arc-files

# 2. train (≈38M parameters, 2k-token examples; tune --bs to your VRAM, 24 GB fits bs 32)
python tools/train_model.py --data data/train/dataset.jsonl --val data/val/dataset.jsonl \
    --out model_big.pt --d 512 --layers 12 --heads 8 --maxlen 2048 --bs 32 --lr 3e-4 \
    --steps 100000 --eval-every 2000 --amp

# 3. evaluate (sample many programs, execution-verified)
python tools/eval_model.py --model model_big.pt --synthetic data/val/dataset.jsonl --n 500 --samples 64 --device cuda
python tools/eval_model.py --model model_big.pt --real challenges.json --samples 128 --device cuda --out real.json

# 4. model + search hybrid (CPU processes, 1 thread each; the model is small enough to sample on CPU too)
python tools/guided_search.py --model model_big.pt --real challenges.json --out g.json \
    --samples 64 --budget 10 --fallback 30 --shard 0/8      # run shards 0..7 in parallel
```

Notes
* `--maxlen 2048` lets ~all real tasks fit (with 1024, 19% of ARC training tasks are skipped because the prompt is too long).
* The 5M CPU model was still improving when it stopped (40k tasks ≈ 1 epoch). Expect the largest gain from data + steps.
* Not implemented yet but recommended for the GPU run: colour-permutation / transpose augmentation of demos,
  re-weighting ops that occur in real tasks (see `examples/real_task_programs.jsonl`), test-time training.
* Resume with `--resume model.pt` (weights only; the LR schedule restarts).
