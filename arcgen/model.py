"""Small GPT that reads demonstration pairs and writes the op sequence (needs ``torch``).

Every grid cell is one token; the embedding also gets the cell's row, column and which grid it
belongs to, so a tiny model can still see 2-D structure. Loss is on the program tokens only.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .serialize import Tokenizer, lex

MAXLEN = 1024  # default; eval/predict read the real limit from the model (cfg['maxlen'] - 128)
N_SEG = 24
NCOORD = 32


# ---- sequence assembly -------------------------------------------------------------

class Encoder:
    """Turns (demos, query, program) into parallel id / row / col / segment arrays."""

    def __init__(self, tok: Tokenizer | None = None):
        self.tok = tok or Tokenizer()
        s = self.tok.stoi
        self.s = s

    def grid(self, g, seg):
        g = np.asarray(g)
        h, w = g.shape
        ids = np.empty(h * (w + 1), dtype=np.int16)
        rows = np.repeat(np.arange(1, h + 1), w + 1).astype(np.int16)
        cols = np.tile(np.arange(0, w + 1), h).astype(np.int16)  # col 0 = the <nl> token
        body = (g.astype(np.int16) + self.s["c0"]).reshape(h, w)
        ids.reshape(h, w + 1)[:, :w] = body
        ids.reshape(h, w + 1)[:, w] = self.s["<nl>"]
        cols = np.minimum(cols, NCOORD - 1)
        rows = np.minimum(rows, NCOORD - 1)
        return ids, rows, cols, np.full(len(ids), min(seg, N_SEG - 1), dtype=np.int16)

    def prompt(self, demos, query):
        s = self.s
        parts = []

        def sp(*names, seg=0):
            n = len(names)
            parts.append((np.array([s[x] for x in names], dtype=np.int16), np.zeros(n, np.int16),
                          np.zeros(n, np.int16), np.full(n, seg, np.int16)))

        sp("<bos>", "<train>")
        seg = 1
        for a, b in demos:
            sp("<in>")
            parts.append(self.grid(a, seg))
            sp("</in>", "<out>")
            parts.append(self.grid(b, seg + 1))
            sp("</out>")
            seg += 2
        sp("<test>", "<in>")
        parts.append(self.grid(query, seg))
        sp("</in>", "<prog>")
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))

    def program(self, text_or_ids):
        ids = text_or_ids if not isinstance(text_or_ids, str) else self.tok.encode_program(text_or_ids)
        ids = list(ids) + [self.s["<eos>"]]
        n = len(ids)
        z = np.zeros(n, np.int16)
        return np.array(ids, np.int16), z, z.copy(), z.copy()

    def example(self, demos, query, program_text):
        p, t = self.prompt(demos, query), self.program(program_text)
        arrs = [np.concatenate([p[k], t[k]]) for k in range(4)]
        return arrs, len(p[0])


class TaskData:
    """Samples training examples from dataset.jsonl records (random demo subsets every epoch)."""

    def __init__(self, path, enc: Encoder, limit=None, seed=0, maxlen=MAXLEN):
        import json
        self.enc, self.rng, self.maxlen = enc, random.Random(seed), maxlen
        self.tasks = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                pairs = [(np.array(p["input"], dtype=np.int8), np.array(p["output"], dtype=np.int8))
                         for p in r["train"] + r["test"]]  # int8: 200k tasks stay ~1.5 GB
                self.tasks.append((pairs, r["program_text"]))

    def __len__(self):
        return len(self.tasks)

    def sample(self):
        while True:
            pairs, text = self.tasks[self.rng.randrange(len(self.tasks))]
            order = list(range(len(pairs)))
            self.rng.shuffle(order)
            k = min(len(pairs) - 1, self.rng.randint(2, 4))
            while k >= 2:
                demos = [pairs[i] for i in order[:k]]
                query = pairs[order[k]][0]
                arrs, n_prompt = self.enc.example(demos, query, text)
                if len(arrs[0]) <= self.maxlen:
                    return arrs, n_prompt
                k -= 1

    def batch(self, bs):
        ex = [self.sample() for _ in range(bs)]
        T = max(len(a[0]) for a, _ in ex)
        ids = torch.zeros(bs, T, dtype=torch.long)
        aux = torch.zeros(3, bs, T, dtype=torch.long)
        mask = torch.zeros(bs, T)
        for b, (arrs, n_prompt) in enumerate(ex):
            n = len(arrs[0])
            ids[b, :n] = torch.from_numpy(arrs[0].astype(np.int64))
            for k in range(3):
                aux[k, b, :n] = torch.from_numpy(arrs[k + 1].astype(np.int64))
            mask[b, n_prompt:n] = 1
        return ids, aux[0], aux[1], aux[2], mask


# ---- model --------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, past=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
        x = x + self.proj(att.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x)), (k, v)


class GPT(nn.Module):
    def __init__(self, vocab, d=256, layers=6, heads=8, maxlen=MAXLEN + 128):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, maxlen=maxlen)
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(maxlen, d)
        self.row, self.col, self.seg = nn.Embedding(NCOORD, d), nn.Embedding(NCOORD, d), nn.Embedding(N_SEG, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids, rows, cols, segs, past=None, start=0):
        T = ids.shape[1]
        pos = torch.arange(start, start + T, device=ids.device)
        x = self.tok(ids) + self.pos(pos) + self.row(rows) + self.col(cols) + self.seg(segs)
        new = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, None if past is None else past[i])
            new.append(kv)
        return self.head(self.ln(x)), new

    @torch.no_grad()
    def sample(self, enc: Encoder, prompt, n=16, temp=0.8, max_new=90, greedy=False):
        """n sampled programs (token id lists) for one prompt tuple from Encoder.prompt."""
        dev = next(self.parameters()).device
        ids, rows, cols, segs = [torch.from_numpy(a.astype(np.int64))[None].to(dev) for a in prompt]
        logits, past = self(ids, rows, cols, segs)
        past = [(k.expand(n, -1, -1, -1).contiguous(), v.expand(n, -1, -1, -1).contiguous()) for k, v in past]
        eos, nxt = enc.s["<eos>"], logits[:, -1].expand(n, -1)
        out = [[] for _ in range(n)]
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        pos = ids.shape[1]
        # only program tokens may be produced: ops, params, punctuation, ints, eos
        allowed = torch.zeros(self.cfg["vocab"], dtype=torch.bool, device=dev)
        lo = enc.s["c0"]
        allowed[:] = True
        allowed[:lo] = False
        allowed[eos] = True
        allowed[lo:lo + 10] = False
        for _ in range(max_new):
            lg = nxt.masked_fill(~allowed, -1e9)
            tok = lg.argmax(-1) if greedy else torch.multinomial(F.softmax(lg / temp, -1), 1)[:, 0]
            for i in range(n):
                if not done[i]:
                    out[i].append(int(tok[i]))
            done |= tok == eos
            if done.all() or pos >= self.cfg["maxlen"] - 1:
                break
            z = torch.zeros(n, 1, dtype=torch.long, device=dev)
            logits, past = self(tok[:, None], z, z, z, past, start=pos)
            nxt, pos = logits[:, -1], pos + 1
        return [o[:-1] if o and o[-1] == eos else o for o in out]


def save(model, path):
    torch.save({"cfg": model.cfg, "state": model.state_dict()}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = GPT(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()
