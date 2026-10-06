# arc-agi-mini — synthetic ARC-AGI task generator

Generates ARC-style tasks (`train`/`test` grid pairs) **together with the exact program**
(a sequence of real grid operations) that maps every input to its output.
Use it to build datasets for models that learn to infer operation sequences from examples.

```
pip install numpy pytest
python -m arcgen list-ops                       # the operation bank (108 ops)
python -m arcgen show --n 3 --max-len 3         # look at a few tasks
python -m arcgen generate --n 10000 --out data --seed 0
```

Output in `--out`:

| file | content |
|---|---|
| `dataset.jsonl` | one record per task: `train`, `test`, `program` (JSON), `program_text`, `ops`, `categories` (+ `traces` with `--trace`: input → after each step → output) |
| `tasks/<id>.json` | plain ARC-AGI format `{"train":[{"input","output"}], "test":[...]}` |
| `stats.json` | op / program-length histogram |

Options: `--min-len/--max-len` (program length 1–4), `--include`, `--exclude`, `--categories geometry,color,object,line,structure`, `--trace`, `--no-arc-files`.
Generation is deterministic per `(seed, index)`.

Program example: `recolor_objects(sel={by=largest},seg=c8,color=3) | flip(axis=1)`

## How it works

* **Layers/objects** (`arcgen/core.py`): a grid is split into colour layers and connected-component objects
  (`seg` = `c4|c8` per-colour, `m4|m8` multicolour). Object ops take a declarative **selector**
  (`largest`, `color`, `size_gt`, `touches_border`, `has_hole`, `rect`, …), edit only those objects and recompose.
* **Operation bank** (`arcgen/ops/`): geometry (rot, flip, scale, tile, symmetrize, half-logic, fractal…),
  colour (recolor, swap, shift, invert, majority…), object (recolor/move/slide/flip/outline/frame/hollow/fill-holes/dilate/erode/crop/count…),
  line (rays, connect pairs, fill lines, stamp shapes), structure (select/overlay cells of separator-split grids)
  and **relation** ops (below).
  Add one with `@op(name, category, sampler)` — the sampler proposes parameters from a probe grid.
* **Sampling** (`arcgen/program.py`): ops are chosen and parametrised step by step against a probe grid so
  parameters (colours, selectors) are meaningful; then many inputs are drawn from an op-matched generator
  (`arcgen/generators.py`) and validated: every step must change most pairs, output ≠ input, outputs not all identical.

## Training data for a model (serialization + tokenizer)

Two ways to train a model that reads demonstration pairs and emits the op sequence:

```
python -m arcgen export --data data/dataset.jsonl --out data/sft.jsonl                    # text: {"prompt","completion"} for any LLM
python -m arcgen export --data data/dataset.jsonl --out data/sft_tok.jsonl --format tokens # ids for the closed-vocab tokenizer
python -m arcgen export ... --target output                                                # predict the output grid directly instead
```

* Every test pair becomes one example: prompt = all `train` pairs + that test input, target = program text.
* `arcgen.serialize.Tokenizer` — closed vocabulary (~256 tokens: `c0..c9` grid cells, `<nl>`, op names,
  parameter names, enum values, ints −30..60). Prompts average ~1k tokens (p95 ≈ 2.2k).
* `parse_program(text)` recovers an executable program from model output (raises `ValueError` if malformed);
  `check_program(text, pairs)` runs it on the demonstrations — use it to filter samples, do best-of-N
  sampling or rejection-based self-training, since a wrong program is detected by execution.

## ARC-2 style relational / context-dependent rules

`arcgen/ops/relations.py` — the answer for a cell/object depends on something else in the grid:

| op | rule |
|---|---|
| `recolor_from_marker` | object touching a 1-cell marker takes its colour |
| `recolor_contained` | objects inside a frame take the frame's colour (or vice versa) |
| `stamp_template` | copy the template object onto every marker pixel |
| `slide_toward` / `connect_to_target` | fly / draw a line towards the target object the object faces |
| `repair_tiling` / `repair_symmetry` | fill a masked patch from the implied period / symmetry (or output only the patch) |
| `extend_periodic` | continue a periodic prefix of each row/column to the border |
| `fill_holes_by_area`, `flood_from_seed` | colour enclosed regions by their size / paint-bucket from seed pixels |
| `draw_rect_between` | two same-colour pixels are opposite corners of a rectangle |
| `fill_largest_empty_rect`, `fill_cells`, `mirror_over_line`, `object_histogram`, `unify_multicolor` | more structure/count rules |

Object ops also gained relational **selectors**: `leftmost/rightmost/topmost/bottommost`,
`touching(colour)`, `common_color/rare_color`, `dup_shape/unique_shape`, `square`, `symmetric`.
(`crop_to_object` + `unique_shape` = "crop the odd one out"; `recolor_objects` + `dup_shape` = "mark repeated shapes".)
`outline_objects`/`frame_objects` accept colour `-1` = the object's own colour.

### Measuring against real ARC data

Two tools, both take an ARC `*_challenges.json` (a dict of `{id: {train, test}}`):

```
python tools/coverage.py   challenges.json            # single op, random parameters (fast, ~5 min)
python tools/search_cov.py challenges.json --out r.json --budget 40   # target-guided beam search, depth 3 (~35 min, 4 cores)
python tools/peek.py challenges.json r.json 12        # print unsolved tasks side by side (find what the bank lacks)
```

`arcgen/search.py` is a beam search that keeps the programs whose outputs are closest to the targets
(cell-wise error; colour parameters are biased towards colours of the target, selectors are enumerated from the
real objects). A task counts as *covered* if the program reproduces **every train output** — the tests' outputs are
not in the file, so this is a lower bound on expressiveness and can include programs that overfit the train pairs.

ARC-AGI-2 training set (1000 tasks):

| bank | single op | beam search |
|---|---|---|
| 55 ops | 6.2% | – |
| 71 ops (+relational) | 7.4% | 9.1% |
| 90 ops (+gap-driven) | – | 10.9% |
| 97 ops | – | 11.8% |
| 108 ops (+round 2) | – | 12.1% (union of 4 stochastic runs: 151 tasks = 15.1%) |

`examples/real_task_programs.jsonl` holds the programs found (union of all runs) (train-verified) — real-task annotations in the
same format as the synthetic dataset. The search is stochastic and time-budgeted, so runs differ by a few tasks.

Workflow used to grow the bank: run the search → look at the unsolved tasks with the smallest residual error
(`tools/peek.py`) → add the missing concept as an op + generator + test → re-measure.

## Synthetic data → model → real tasks (end-to-end)

```
pip install numpy torch
python -m arcgen generate --n 40000 --out data/train --seed 100 --workers 4 --no-arc-files   # ~5 min
python -m arcgen generate --n 500   --out data/val   --seed 200 --workers 4 --no-arc-files
python tools/train_model.py --data data/train/dataset.jsonl --val data/val/dataset.jsonl --out model.pt --steps 5000
python tools/eval_model.py  --model model.pt --synthetic data/val/dataset.jsonl --n 200      # held-out synthetic
python tools/eval_model.py  --model model.pt --real challenges.json --out real.json          # real ARC tasks
```

`arcgen/model.py`: a 5M-parameter GPT (6 layers, d=256). Each grid cell is a token that also carries
row / column / which-grid embeddings; the prompt is the demonstration pairs plus a query input and the target is
the program text (loss on the program only). Every epoch re-samples which pairs are demos and which is the query.
Sampling uses a KV cache and only allows program tokens; every sampled program is parsed and **executed on the
demonstrations** — only programs that reproduce all of them count.

### Results of the first model (5M params, 40k synthetic tasks, 5000 steps ≈ 2 h on 4 CPU cores)

| test | result |
|---|---|
| validation token accuracy (teacher forced) | 0.80 |
| held-out synthetic tasks, 32 samples + greedy, program reproduces all demos | 9.5% (1-op tasks 21%, 2-op 6%, 3-op 8%, 4-op 0%) |
| … and also reproduces the held-out test pair | 7.8% |
| syntactically valid sampled programs | 95% |
| **real ARC-AGI-2 training tasks**, train-verified | **23 / 1000** (2.3%); 190 tasks skipped because the prompt exceeds 924 tokens |
| of those, also found by beam search / new | 19 / 4 |

The model clearly learns the format and the simple, visually obvious ops (fractal, mirror/tile, half-logic, rot_quad,
scale, periods) but is far from solving ARC: greedy decoding is almost never right (0.3%), sampling + execution
check is what produces the hits. Real-task numbers are *train-verified only* (test outputs are not available),
so a few programs may be spurious (e.g. no-op steps such as `swap_colors(a=3,b=3)`).
`examples/model_real_solutions.jsonl` lists the programs; `models/gpt5m_synthetic.pt` is the checkpoint
(`arcgen.model.load`).

Obvious next levers (not done): more/larger data and longer training (loss was still falling), a bigger model / GPU,
dropping no-op steps from the sampler, sampling more candidates (best-of-256), and combining the model's
proposals with the beam search (use sampled programs as a prior / starting points for the search).

## Model + search hybrid (`arcgen/guided.py`, `tools/guided_search.py`)

The model only has to say *which ops*; the beam search finds *which parameters*. Per task: sample 48 programs →
(1) a sample that already reproduces the demos, (2) fix the most frequent **op skeletons** and search their
parameters, (3) search restricted to the model's ops, (4) fall back to unrestricted beam search.
Evaluated on a fixed subset of 125 real tasks (every 8th), ~40 s of CPU per task in total:

| method | solved (train-verified) |
|---|---|
| model alone (32 samples) | 6 |
| unrestricted beam search, 3 independent runs | 16, 20, 20 |
| model-restricted search only (20 s) | 16 (on the 102 tasks whose prompt fit; beam: 20) |
| **hybrid** (10 s model-guided + 28 s beam) | **22** |
| hybrid ∪ one beam run | 23 |

Take-aways: restricting the search to the model's guesses *hurts* (a weak model excludes the right op); using the
guesses as a cheap first stage in front of the full search gives a small gain (+2 to +6 tasks of 125, i.e. a few
points — with n=125 this is suggestive, not conclusive). The gain should grow with a better model:
see `docs/GPU_TRAINING.md`. Solutions: `examples/guided_subset_solutions.jsonl`.

## Hand-designed task families (`arcgen/curated.py`) and the notebook

Random op sampling rarely yields puzzles whose second step depends on what the first one *introduced*. The 33
**families** are small stories with linked parameters, a matching input generator and an English description, e.g.
*"Paint the largest object green, then cut it out"*, *"Erase speckles, then flood every closed ring"*,
*"Repair the holes in the carpet, then output its smallest repeating motif"*, *"Objects fly to the wall, then glow"*.

```
python -m arcgen families                                                        # list them
python -m arcgen generate --kind curated --n 50000 --out data/cur --no-arc-files   # only families
python -m arcgen generate --kind mixed   --n 200000 --out data/train --workers 12 --no-arc-files   # 50/50 with random sampling
```
Records get extra fields `family`, `story`, `tags`. Add your own with the `@family(...)` decorator (see the notebook, section 4).

`notebooks/arc_synthetic_playground.ipynb` (Colab / Kaggle / local) walks through: operation bank → browse puzzles with
step-by-step solutions → write a new family → build a mixed dataset → train → per-token diagnosis → execution-verified
evaluation → model+search hybrid on a held-out (or real) task. `QUICK = True` runs the whole thing in ~1 minute on CPU.
Whether curated data helps the model is an open question: compare `--kind random` vs `--kind mixed` at equal size on
`tools/eval_model.py --real` before committing to a long run.

## Reality check on ARC-AGI-2 evaluation (important)

On the public **evaluation** tasks the 108-op bank fits **0 of 120** tasks (demos only) and the fail-closed Kaggle branch
accepts 0 of 86 (training tasks: 12-15 % and 10.8 % with 93 % precision). The Kaggle run with the branch scored the same
as without it. The synthetic data / model / search machinery is therefore only useful as *infrastructure* (e.g. pre-training
data for a neural model), not as a stand-alone solver for ARC-AGI-2. `tools/analyze_candidate_pool.py` measures how much
headroom a better answer *ranker* has on top of the Qwen pipeline.
