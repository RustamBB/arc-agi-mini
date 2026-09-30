"""Model-guided search: the model proposes *which ops*, the search finds *which parameters*."""
from __future__ import annotations

import re
import time
from collections import Counter

from .ops import OPS
from .program import to_text
from .search import search
from .serialize import check_program

_STEP = re.compile(r"\s*([a-z_0-9]+)\(")


def skeleton(text: str):
    """Op names of a (possibly malformed) program text, or None if a name is unknown."""
    names = []
    for step in text.split("|"):
        m = _STEP.match(step)
        if not m or m.group(1) not in OPS:
            return None
        names.append(m.group(1))
    return tuple(names) if 1 <= len(names) <= 4 else None


def guided_solve(pairs, texts, budget=20.0, n_skel=12, seed=0):
    """``texts``: programs sampled from the model. Returns (program or None, how) where ``how`` is
    'model' (a sample already verified), 'skeleton' (fixed op sequence, searched parameters) or 'ops' (search
    restricted to the ops the model used)."""
    for t in texts:
        if check_program(t, pairs):
            from .serialize import parse_program
            return parse_program(t), "model"
    skels = Counter(s for s in map(skeleton, texts) if s)
    if not skels:
        return None, None
    t0 = time.time()
    ranked = [s for s, _ in skels.most_common(n_skel)]
    per = max(1.0, budget * 0.6 / len(ranked))
    for k, sk in enumerate(ranked):
        prog, _, _ = search(pairs, beam=4, tries=120, top=24, budget=per, seed=seed + k,
                            ops_by_depth=[[n] for n in sk])
        if prog and check_program(to_text(prog), pairs):
            return prog, "skeleton"
    op_count = Counter(n for s in skels.elements() for n in s)
    pool = [n for n, _ in op_count.most_common(25)]
    left = max(1.0, budget - (time.time() - t0))
    prog, _, _ = search(pairs, depth=3, beam=4, tries=100, top=24, budget=left, seed=seed + 99, ops=pool)
    if prog and check_program(to_text(prog), pairs):
        return prog, "ops"
    return None, None
