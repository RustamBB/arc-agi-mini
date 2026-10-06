# arcnet — a from-scratch layer / object / relation model with a LADDER curriculum

Goal (as specified): decompose a grid into **layers**, find **relations between shapes**, and make equivalent shapes —
a rectangle or any complicated figure, wherever it is, however it is rotated or coloured — come out as **one class / one
number**. Trained from scratch (no pretrained weights), TRM-style recursion, a per-task embedding that is trained on the
task's own demonstration pairs (test-time training; evaluation solutions are used for scoring only).

## Architecture (`arcnet/features.py`, `arcnet/model.py`)
* **Tokens**: 2 task tokens + 10 colour-layer tokens + K object tokens + G·G cell tokens (G=30, K=64).
* **Shape classes (the "one class / one number")**: every object gets a translation-normalised mask, then
  * exact shape, and shape **up to rotation + reflection (D4)** — an L, its mirror image and a rotated, recoloured L share one id;
  * size, dims, dims up to rotation, number of holes, colour — six equivalence notions at once, because tasks differ in which sameness matters;
  * **numbers**: how many objects share my class (per notion), size rank (largest/smallest), holes, layer counts.
* **Relations** (object↔object, 14): same shape / same shape up to D4 / same size / same dims / same dims rotated / same
  holes / same colour / touches / contains / inside / row-aligned / column-aligned / left-of / above; layer↔layer (5).
  They enter attention as learned per-head biases (plus relative 2-D position bias, same-object and same-colour for cells).
* **ClassPool**: after each pass, objects equivalent under a notion average their states (gated per notion), so
  equivalent shapes receive identical information. Every cell also receives the embedding of *its* object and *its*
  colour layer directly.
* **Recursion**: a weight-shared block stack is looped (default 6×4 layers); the soft answer is fed back (detached);
  every loop is supervised. Output: 30×30 cell classes with a PAD class that encodes the output size.
* **Augmentation done right**: each task has A fixed augmentations (dihedral × colour permutation) and **each
  (task, augmentation) pair has its own embedding**. (Independent random augmentation per sample with a shared task
  embedding makes any colour- or direction-specific rule unlearnable — found and fixed during testing.) Inference votes
  over the task's A augmentations.

## LADDER (`arcnet/ladder.py`) — Simonds & Yoneda 2025, adapted
LADDER = the model recursively builds *simpler variants of a hard problem*, learns on them with a verifiable reward and,
at test time, learns on variants of the test problem (TTRL). Here the ladder is **compositional** (`--ladder-mode
compositional`, default): difficulty = the number of units (colour layers or objects) that are present, k = 1, 2, 3, 4.
* **One layer alone -> a layer with a second one -> with a third ...** (`--ladder-order pairs`): unit i together with each
  other unit *separately*, then every unit takes the base role, then larger combinations; **or cumulative**
  (`cumulative`): U1, U1+U2, U1+U2+U3, ... along a random order; `both` (default) uses the two. Units are colour layers
  and, separately, multicolour objects.
* **Verified labels** (`restrict`): kept input cells define the variant; erased cells' changes disappear from the output;
  newly drawn cells are grouped into components and attributed to the kept or erased unit they touch. A variant is
  **rejected** when units interact (a changed cell next to an erased cell, a drawn component touching both a kept and an
  erased unit, a floating drawn component) or when the rule is no longer visible / shows a colour transition the original
  pair does not. Example: "recolour 1-objects that touch colour 2" never yields layer 1 without layer 2.
* **Difficulty-driven schedule**: per (task, level k) the model's own exact-match is tracked (EMA); sampling weight peaks at the
  frontier (success ≈ 0.5), so each task is climbed from few units to the full grid.
* **TTRL phase** (`--ttrl-steps`): the same schedule restricted to the evaluation tasks' demos and their variants.
* Coverage on the real files: training 392/1000 tasks get variants, **evaluation 59/120** (the random keep-fraction/crop ladder,
  `--ladder-mode fractional`, reaches 437/1000 and only 35/120).
* Differences from the paper: supervised cross-entropy instead of GRPO; variants come from rules, not from the model;
  only same-shape tasks get variants (~68 % of the tasks).

## What has actually been verified (CPU, no GPU/cloud run yet)
* Shape classes are invariant to rotation/reflection/colour/translation (tests/test_arcnet.py).
* **Shape-equivalence sanity task** (`tools/arcnet_sanity.py`): recolour the objects whose shape occurs more than once
  (task 0) / only once (task 1), shapes in random orientations, colours and places. A 64-dim, 2-layer model reaches
  **100 % exact match on 128 fresh samples** after ~700 steps (≈4 min on CPU) for both rules.
* LADDER variants keep a known rule intact; the schedule prefers the frontier; the full CLI (ladder + TTRL + evaluation)
  runs on the real ARC files with a tiny model.
* **Not verified**: any ARC-AGI-2 accuracy. Literature for comparable from-scratch recursive models (TRM) is ≈8 % on
  ARC-AGI-2 public eval (HRM ≈5 %), far below the Qwen3-4B + TTT pipeline; the realistic role of this model is an extra,
  *different* candidate generator for an ensemble. Whether the relation/class machinery and LADDER improve on that is
  exactly what the cloud run should measure (ablate with the flags below).

## Cloud run (A100/H100)
```bash
pip install numpy torch
python -m arcnet.train \
  --train-challenges arc-agi_training_challenges.json \
  --eval-challenges  arc-agi_evaluation_challenges.json --eval-solutions arc-agi_evaluation_solutions.json \
  --out arcnet.pt --d 384 --layers 4 --loops 6 --heads 8 --aug 16 --bs 16 --lr 3e-4 --amp \
  --steps 100000 --ladder --ladder-mode compositional --ladder-order both --ttrl-steps 30000 --workers 16 --eval-every 5000
```
Memory/speed: attention is dense over ≈976 tokens with a [B,H,T,T] bias (≈0.5 GB at B=32); expect on the order of
100 samples/s on an A100 (my estimate, unmeasured). Ablations to run (same budget): `--ladder` off / `fractional` / `compositional`, `--ladder-order cumulative|pairs|both`, `--ttrl-steps 0`,
`--aug 1`. Eval credit is printed every `--eval-every` steps (task credit with 2 attempts over the eval demos-trained IDs).

## Trying it on Kaggle / Colab without any pre-training (`notebooks/arcnet_kaggle.ipynb`)
arcnet needs no pre-trained weights: each task's rule is learned from the task's own demos (per-(task, augmentation)
embedding), so "just try it" means *training inside the notebook for a fixed time*. An **untrained** model is random.
```
Kaggle: add the ARC Prize 2026 competition data, turn a GPU on, Run All.  (no internet needed, the code is embedded)
Colab / local: put arc-agi_training_challenges.json, arc-agi_evaluation_challenges.json (+ arc-agi_evaluation_solutions.json
               to score) in a folder and set ARC_ROOT=<folder>.
ARCNET_MINUTES=60 (default) is the whole training budget (70 % main phase, 30 % test-time phase on the evaluation tasks);
ARCNET_QUICK=1 is a 1-minute CPU smoke test.
```
It prints the public-evaluation credit (2 attempts), writes `arcnet_candidates.json` (top-8 grids + scores per test input),
`submission_arcnet.json` and `arcnet.pt`, and — if `qwen_candidate_pool.json` is present — how many queries arcnet gets
right that the Qwen hybrid misses (the only reason to ensemble it). Verified here: the whole notebook runs end-to-end on the
real ARC files (tiny CPU config); data loading is ~210 samples/s per CPU worker (4.7 ms), enough for a GPU. **Not
verified**: GPU speed/memory, the fp16 path used on T4/P100 (GradScaler), and any score — expect low numbers after an hour.
Rebuild after code changes: `python tools/build_arcnet_notebook.py`.
