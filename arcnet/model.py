"""Layer-Relation Recursive Model (LRRM).

Tokens:  [task tokens] + [10 colour-layer tokens] + [K object tokens] + [G*G cell tokens]
Attention is biased by explicit relations: cell-cell (relative 2-D position, same object, same colour),
cell<->object (membership, same colour), cell<->layer, object<->object (14 relations incl. same shape up to D4, same
size/dims/holes/colour, touches, contains, aligned...), layer<->layer, object<->layer.
After every pass, **ClassPool** averages the states of objects that are equivalent under each notion (same shape up to
D4, same size, ...), gated per notion, so equivalent shapes receive the same information -> the same class/number.
A weight-shared stack is looped (TRM-style recursion) with the current soft answer fed back; every loop is supervised.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import EQUIV, MAX_MULT, N_LAYER_ATTR, N_OBJ_ATTR, PAD, REL_LL, REL_OO

NEG = -1e4


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bias):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.n2(x))


class ClassPool(nn.Module):
    """Objects equivalent under notion e share information: h += sum_e gate_e(h) * W_e(mean_{o' ~_e o} h(o'))."""

    def __init__(self, d, n_equiv):
        super().__init__()
        self.gate = nn.Linear(d, n_equiv)
        self.w = nn.ModuleList(nn.Linear(d, d) for _ in range(n_equiv))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, cls):                      # h [B,K,d]; cls [B,E,K] (-1 invalid)
        g = torch.sigmoid(self.gate(self.norm(h)))  # [B,K,E]
        hn = self.norm(h)
        out = 0
        for e, w in enumerate(self.w):
            c = cls[:, e]
            same = (c[:, :, None] == c[:, None, :]) & (c[:, :, None] >= 0) & (c[:, None, :] >= 0)
            m = same.to(hn.dtype)
            m = m / m.sum(-1, keepdim=True).clamp(min=1)
            out = out + g[..., e:e + 1] * w(m @ hn)
        return h + out


class LRRM(nn.Module):
    def __init__(self, n_tasks, G=30, K=64, d=256, heads=8, layers=4, loops=6, task_tokens=2, A=1):
        super().__init__()
        self.cfg = dict(n_tasks=n_tasks, G=G, K=K, d=d, heads=heads, layers=layers, loops=loops,
                        task_tokens=task_tokens, A=A)
        self.A = A
        self.G, self.K, self.d, self.H, self.loops, self.TT = G, K, d, heads, loops, task_tokens
        self.Tc = G * G
        self.T = task_tokens + 10 + K + self.Tc
        self.o_task, self.o_layer, self.o_obj = 0, task_tokens, task_tokens + 10
        self.o_cell = task_tokens + 10 + K
        # embeddings
        self.col = nn.Embedding(11, d)
        self.row_e, self.col_e = nn.Embedding(G + 1, d), nn.Embedding(G + 1, d)
        self.ans = nn.Linear(11, d, bias=False)
        self.task = nn.Embedding(n_tasks * A, task_tokens * d)   # one embedding per (task, augmentation)
        self.task_cond = nn.Linear(task_tokens * d, d)            # task vector added to EVERY token (global conditioning)
        self.o_size, self.o_h, self.o_w = nn.Embedding(64, d), nn.Embedding(G + 1, d), nn.Embedding(G + 1, d)
        self.o_holes, self.o_rect = nn.Embedding(8, d), nn.Embedding(2, d)
        self.o_rank_d, self.o_rank_a = nn.Embedding(MAX_MULT + 1, d), nn.Embedding(MAX_MULT + 1, d)
        self.o_mult = nn.ModuleList(nn.Embedding(MAX_MULT + 1, d) for _ in EQUIV)   # 'how many share my class'
        self.l_col = nn.Embedding(10, d)
        self.l_pres, self.l_cnt, self.l_nobj = nn.Embedding(2, d), nn.Embedding(64, d), nn.Embedding(32, d)
        self.l_h, self.l_w = nn.Embedding(G + 1, d), nn.Embedding(G + 1, d)
        # attention-bias parameters
        R = 2 * G - 1
        self.pos_bias = nn.Parameter(torch.zeros(heads, R, R))
        self.w_cc = nn.Parameter(torch.zeros(2, heads))          # same object, same colour
        self.w_co = nn.Parameter(torch.zeros(2, heads))          # cell->object: member, same colour
        self.w_oc = nn.Parameter(torch.zeros(2, heads))
        self.w_cl = nn.Parameter(torch.zeros(heads))
        self.w_lc = nn.Parameter(torch.zeros(heads))
        self.w_ol = nn.Parameter(torch.zeros(heads))
        self.w_lo = nn.Parameter(torch.zeros(heads))
        self.oo = nn.Linear(len(REL_OO), heads, bias=False)
        self.ll = nn.Linear(len(REL_LL), heads, bias=False)
        ar = torch.arange(G)
        r, c = ar.repeat_interleave(G), ar.repeat(G)                # row / col of every cell token
        dr = r[:, None] - r[None, :] + G - 1
        dc = c[:, None] - c[None, :] + G - 1
        self.register_buffer("rel_index", dr * R + dc, persistent=False)   # [Tc,Tc] -> relative-position bias slot
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.pool = ClassPool(d, len(EQUIV))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 11)
        self.apply(self._init)
        with torch.no_grad():                                   # sensible priors: look at my own object / colour layer
            self.w_cc[0].fill_(2.0); self.w_co[0].fill_(4.0); self.w_oc[0].fill_(4.0)
            self.w_cl.fill_(2.0); self.w_lc.fill_(2.0); self.w_ol.fill_(2.0); self.w_lo.fill_(2.0)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    # ---- input embedding -------------------------------------------------------------------------------
    def embed(self, b):
        B, G, K = b["color"].shape[0], self.G, self.K
        dev = b["color"].device
        ar = torch.arange(G, device=dev)
        color = b["color"].view(B, self.Tc)
        rows = ar.repeat_interleave(G)[None]
        cols = ar.repeat(G)[None]
        cell = self.col(color) + self.row_e(rows) + self.col_e(cols)
        a = b["obj_attr"]
        valid = a[..., 0].float().unsqueeze(-1)
        obj = (self.col(a[..., 1]) + self.o_size(a[..., 2]) + self.o_h(a[..., 3]) + self.o_w(a[..., 4])
               + self.o_holes(a[..., 5]) + self.o_rect(a[..., 6]) + self.row_e(a[..., 7]) + self.col_e(a[..., 8])
               + self.o_rank_d(a[..., 9]) + self.o_rank_a(a[..., 10])
               + sum(emb(b["obj_mult"][..., e]) for e, emb in enumerate(self.o_mult))) * valid
        la = b["layer_attr"]
        lay = (self.l_col.weight[None] + self.l_pres(la[..., 0]) + self.l_cnt(la[..., 1]) + self.l_nobj(la[..., 2])
               + self.l_h(la[..., 3]) + self.l_w(la[..., 4]))
        # hard-wired hierarchy: every cell also receives the embedding of ITS object (shape class, multiplicity,
        # size rank ...) and of ITS colour layer, so "what kind of shape am I part of" is available from layer 0
        oi = b["obj_idx"].view(B, self.Tc)
        own = torch.gather(obj, 1, oi.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d)) * (oi >= 0).unsqueeze(-1)
        lay_of = torch.gather(lay, 1, color.clamp(max=9).unsqueeze(-1).expand(-1, -1, self.d)) * (color < 10).unsqueeze(-1)
        tv = self.task(b["task_id"])
        cond = self.task_cond(tv).unsqueeze(1)
        task = tv.view(B, self.TT, self.d)
        return torch.cat([task, lay, obj, cell + own + lay_of], 1) + cond

    # ---- relation-biased attention bias ---------------------------------------------------------------------
    def make_bias(self, b):
        B, H, K, Tc, T = b["color"].shape[0], self.H, self.K, self.Tc, self.T
        dev = b["color"].device
        bias = torch.zeros(B, H, T, T, device=dev)
        color = b["color"].view(B, Tc)
        oi = b["obj_idx"].view(B, Tc)
        a = b["obj_attr"]
        ar10 = torch.arange(10, device=dev)
        # cell-cell
        pos = self.pos_bias.view(H, -1)[:, self.rel_index]                       # [H,Tc,Tc]
        so = ((oi[:, :, None] == oi[:, None, :]) & (oi[:, :, None] >= 0)).float()
        sc = (color[:, :, None] == color[:, None, :]).float()
        cc = pos[None] + self.w_cc[0].view(1, H, 1, 1) * so[:, None] + self.w_cc[1].view(1, H, 1, 1) * sc[:, None]
        c0 = self.o_cell
        bias[:, :, c0:, c0:] = cc
        # cell <-> object
        mem = (oi[:, :, None] == torch.arange(K, device=dev)[None, None]).float()           # [B,Tc,K]
        scol = (color[:, :, None] == a[:, None, :, 1]).float()
        o0 = self.o_obj
        bias[:, :, c0:, o0:o0 + K] = self.w_co[0].view(1, H, 1, 1) * mem[:, None] + self.w_co[1].view(1, H, 1, 1) * scol[:, None]
        bias[:, :, o0:o0 + K, c0:] = (self.w_oc[0].view(1, H, 1, 1) * mem.transpose(1, 2)[:, None]
                                      + self.w_oc[1].view(1, H, 1, 1) * scol.transpose(1, 2)[:, None])
        # cell <-> layer
        l0 = self.o_layer
        cl = (color[:, :, None] == ar10[None, None]).float()                                # [B,Tc,10]
        bias[:, :, c0:, l0:l0 + 10] = self.w_cl.view(1, H, 1, 1) * cl[:, None]
        bias[:, :, l0:l0 + 10, c0:] = self.w_lc.view(1, H, 1, 1) * cl.transpose(1, 2)[:, None]
        # object <-> object, layer <-> layer, object <-> layer
        bias[:, :, o0:o0 + K, o0:o0 + K] = self.oo(b["obj_rel"].float()).permute(0, 3, 1, 2)
        bias[:, :, l0:l0 + 10, l0:l0 + 10] = self.ll(b["layer_rel"].float()).permute(0, 3, 1, 2)
        ol = (a[:, :, 1, None] == ar10[None, None]).float() * a[:, :, 0, None].float()          # [B,K,10]
        bias[:, :, o0:o0 + K, l0:l0 + 10] = self.w_ol.view(1, H, 1, 1) * ol[:, None]
        bias[:, :, l0:l0 + 10, o0:o0 + K] = self.w_lo.view(1, H, 1, 1) * ol.transpose(1, 2)[:, None]
        # invalid objects cannot be attended to
        kmask = torch.zeros(B, T, device=dev)
        kmask[:, o0:o0 + K] = (1 - a[..., 0].float()) * NEG
        return bias + kmask[:, None, None, :]

    # ---- forward -------------------------------------------------------------------------------------------
    def forward(self, b, return_all=True):
        x0 = self.embed(b)
        bias = self.make_bias(b)
        B = x0.shape[0]
        h = x0
        a = torch.zeros(B, self.Tc, self.d, device=x0.device, dtype=x0.dtype)
        outs = []
        o0, K, c0 = self.o_obj, self.K, self.o_cell
        for _ in range(self.loops):
            z = h + x0
            z = torch.cat([z[:, :c0], z[:, c0:] + a], 1)
            for blk in self.blocks:
                z = blk(z, bias)
            z = torch.cat([z[:, :o0], self.pool(z[:, o0:o0 + K], b["obj_cls"]), z[:, o0 + K:]], 1)
            h = z
            logits = self.head(self.norm(h[:, c0:]))
            outs.append(logits)
            a = self.ans(F.softmax(logits.detach().float(), -1)).to(x0.dtype)
        return outs if return_all else outs[-1]


def loss_fn(outs, target, weights=None):
    """target [B,G,G] in 0..10 (10 = PAD). Cells inside the target rectangle weigh 1, PAD cells 0.1; later loops more."""
    B = target.shape[0]
    t = target.view(B, -1)
    inside = (t != PAD).float()
    w = inside + 0.1 * (1 - inside)
    ws = weights or [(i + 1) for i in range(len(outs))]
    tot = 0
    for wi, lg in zip(ws, outs):
        ce = F.cross_entropy(lg.float().reshape(-1, 11), t.reshape(-1), reduction="none").view(B, -1)
        tot = tot + wi * (ce * w).sum() / w.sum()
    return tot / sum(ws)
