# arc-agi-mini — synthetic ARC-AGI task generator

Generates ARC-style tasks (`train`/`test` grid pairs) **together with the exact program**
(a sequence of real grid operations) that maps every input to its output.
Use it to build datasets for models that learn to infer operation sequences from examples.

```
pip install numpy pytest
python -m arcgen list-ops                       # the operation bank (55 ops)
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
  line (rays, connect pairs, fill lines, stamp shapes) and structure (select/overlay cells of separator-split grids).
  Add one with `@op(name, category, sampler)` — the sampler proposes parameters from a probe grid.
* **Sampling** (`arcgen/program.py`): ops are chosen and parametrised step by step against a probe grid so
  parameters (colours, selectors) are meaningful; then many inputs are drawn from an op-matched generator
  (`arcgen/generators.py`) and validated: every step must change most pairs, output ≠ input, outputs not all identical.
