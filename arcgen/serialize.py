"""Text serialization, closed-vocabulary tokenizer and program parser.

Canonical program text (== ``program.to_text``)::

    recolor_objects(sel={by=color,value=3},seg=c8,color=5) | flip(axis=1)

* ``parse_program(text)`` turns model output back into an executable program.
* ``check_program(text, pairs)`` runs it on demonstration pairs (verification / search).
* ``Tokenizer`` maps prompts (demonstrations + query) and targets to ids with a closed
  vocabulary: grid cells ``c0..c9``, ``<nl>``, op names, parameter names, enum values, ints.
"""
from __future__ import annotations

import inspect
import re

import numpy as np

from .core import OpError
from .ops import OPS
from .ops.base import _SEL
from .ops.lines import PATTERNS, RAYS
from .ops.objects import DIRS
from .program import run, same, to_text

_LEX = re.compile(r"\s*(-?\d+|[A-Za-z_][A-Za-z_0-9]*|[(){}\[\],=|])")


def lex(text: str) -> list[str]:
    toks, pos, text = [], 0, text.strip()
    while pos < len(text):
        m = _LEX.match(text, pos)
        if not m:
            raise ValueError(f"bad character at {pos}: {text[pos:pos + 10]!r}")
        toks.append(m.group(1))
        pos = m.end()
    return toks


# ---- parser ---------------------------------------------------------------------

def _value(t, i):
    tok = t[i]
    if tok == "{":
        d, i = {}, i + 1
        while t[i] != "}":
            k = t[i]
            assert t[i + 1] == "=", "expected ="
            d[k], i = _value(t, i + 2)
            if t[i] == ",":
                i += 1
        return d, i + 1
    if tok == "[":
        lst, i = [], i + 1
        while t[i] != "]":
            v, i = _value(t, i)
            lst.append(v)
            if t[i] == ",":
                i += 1
        return lst, i + 1
    if re.fullmatch(r"-?\d+", tok):
        return int(tok), i + 1
    if tok in ("true", "false"):
        return tok == "true", i + 1
    return tok, i + 1


def parse_program(text: str) -> list[dict]:
    """Parse canonical program text; raises ValueError on malformed / unknown programs."""
    try:
        t, i, prog = lex(text), 0, []
        while i < len(t):
            name = t[i]
            if name not in OPS or t[i + 1] != "(":
                raise ValueError(f"bad op {name!r}")
            i += 2
            params = {}
            while t[i] != ")":
                k = t[i]
                assert t[i + 1] == "=", "expected ="
                params[k], i = _value(t, i + 2)
                if t[i] == ",":
                    i += 1
            i += 1
            prog.append({"op": name, "params": params})
            if i < len(t):
                assert t[i] == "|", "expected |"
                i += 1
        if not prog:
            raise ValueError("empty program")
        return prog
    except (IndexError, AssertionError) as e:
        raise ValueError(f"malformed program: {e}") from e


def check_program(text: str, pairs) -> bool:
    """True if the program parses and maps every input of ``pairs`` to its output."""
    try:
        prog = parse_program(text)
        return all(same(run(prog, np.asarray(a))[-1], np.asarray(b)) for a, b in pairs)
    except (ValueError, TypeError, KeyError, OpError, IndexError):
        return False


# ---- grid text ------------------------------------------------------------------

def grid_text(g) -> str:
    return "\n".join("".join(str(int(v)) for v in row) for row in g)


def prompt_text(train, query) -> str:
    """Plain-text prompt for off-the-shelf LLM fine-tuning."""
    parts = [f"INPUT\n{grid_text(a)}\nOUTPUT\n{grid_text(b)}\n" for a, b in train]
    return "\n".join(parts) + f"\nINPUT\n{grid_text(query)}\nPROGRAM\n"


# ---- tokenizer ------------------------------------------------------------------

def _param_names() -> set[str]:
    names = {"by", "value"}
    for o in OPS.values():
        names |= set(list(inspect.signature(o.fn).parameters)[1:])
    return names


ENUMS = (set(_SEL) | set(RAYS) | set(PATTERNS) | set(DIRS)
         | {"c4", "c8", "m4", "m8", "h", "v", "both", "rot", "and", "or", "xor", "nor", "a_not_b",
            "most", "least", "top", "bottom", "left", "right", "asc", "desc", "fwd", "rev",
            "true", "false", "inherit", "invert", "minor", "major", "nonempty", "empty"})
SPECIAL = ["<pad>", "<bos>", "<eos>", "<train>", "<test>", "<in>", "</in>", "<out>", "</out>",
           "<prog>", "<nl>"]
PUNCT = list("(){}[],=|")


class Tokenizer:
    def __init__(self):
        vocab = (SPECIAL + [f"c{i}" for i in range(10)] + PUNCT + sorted(OPS)
                 + sorted(_param_names()) + sorted(ENUMS) + [str(i) for i in range(-30, 61)])
        self.itos = list(dict.fromkeys(vocab))  # a name may be both a key and an enum value
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def ids(self, toks):
        return [self.stoi[t] for t in toks]

    # grids
    def grid_tokens(self, g) -> list[str]:
        toks = []
        for row in np.asarray(g):
            toks += [f"c{int(v)}" for v in row] + ["<nl>"]
        return toks

    def decode_grid(self, ids) -> np.ndarray:
        rows, cur = [], []
        for i in ids:
            t = self.itos[i]
            if t == "<nl>":
                rows.append(cur)
                cur = []
            elif t in ("<eos>", "<pad>"):
                break
            elif re.fullmatch(r"c\d", t):
                cur.append(int(t[1]))
            else:
                raise ValueError(f"unexpected token {t}")
        if cur or not rows or len({len(r) for r in rows}) != 1:
            raise ValueError("ragged / empty grid")
        return np.array(rows)

    # programs
    def encode_program(self, text_or_program) -> list[int]:
        text = text_or_program if isinstance(text_or_program, str) else to_text(text_or_program)
        return self.ids(lex(text))

    def decode_program(self, ids) -> str:
        out = []
        for i in ids:
            t = self.itos[i]
            if t in ("<eos>", "<pad>"):
                break
            out.append(t)
        s = " ".join(out)
        s = re.sub(r"\s*([(){}\[\],=])\s*", r"\1", s)
        return s.replace("|", " | ").replace("  ", " ").strip()

    # prompts
    def encode_prompt(self, train, query) -> list[int]:
        toks = ["<bos>", "<train>"]
        for a, b in train:
            toks += ["<in>"] + self.grid_tokens(a) + ["</in>", "<out>"] + self.grid_tokens(b) + ["</out>"]
        toks += ["<test>", "<in>"] + self.grid_tokens(query) + ["</in>", "<prog>"]
        return self.ids(toks)

    def encode_example(self, rec: dict, test_idx: int = 0, target: str = "program"):
        """(prompt_ids, target_ids) from a dataset.jsonl record.

        target='program' -> op sequence; target='output' -> the query's output grid."""
        train = [(np.array(p["input"]), np.array(p["output"])) for p in rec["train"]]
        q = rec["test"][test_idx]
        prompt = self.encode_prompt(train, np.array(q["input"]))
        if target == "program":
            tgt = self.encode_program(rec["program_text"])
        else:
            tgt = self.ids(self.grid_tokens(np.array(q["output"])))
        return prompt, tgt + [self.stoi["<eos>"]]
