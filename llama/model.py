import mlx
import mlx.nn as nn
import mlx.core as mx

import argparse
import glob
import json
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

"""
what i need
- model (LM)
    - RMSNorm (before self attention/ffn)
    - RoPE
    - self attention using RoPE
    - residual connections
- model (vqgan? the image one)
- model configs (vqgan stays the same)
- training code
    - training scheduling as per the recipe in chameleon, then c3mleon
- data preparation

addons later:
- kv cache
"""

@dataclass
class LlamaConfig:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    head_dim: int 
    hidden_dim: int
    n_qhead: int
    n_kvhead: int
    n_embd: int
    dropout: float = 0.0
    bias: bool
    vocab_size: int
    norm_eps: float = 1e-6
    max_seq_len: int
    rope_theta: float = 10000.0


def polar(abs, angle):
    return abs * mx.cos(angle) + abs * mx.sin(angle) 


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = mx.arange(end)
    freqs = mx.outer(t, freqs) 
    freqs_cis = polar(mx.ones_like(freqs), freqs)
    return freqs_cis
    

def repeat_kv(x, n_rep: int):
    bs, slen, n_kvhead, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kvhead, n_rep, head_dim)
            .reshape(bs, slen, n_kvhead * n_rep, head_dim)
    )


def hstack(a, b):
    assert a.shape == b.shape
    if len(a.shape) == 1:
        return mx.concatenate([a, b])
    return mx.concatenate([a, b], axis=1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w1 = nn.Linear(dim, dim) 

    def _norm(self, x):
        return x * mx.rsqrt(mx.power(x, 2).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        return self.w1(self._norm(x.float()))

class SwiGLU(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, bias=True) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dims, out_dims, bias=bias)
        self.w2 = nn.Linear(in_dims, out_dims, bias=bias)

    def __call__(self, x):
        x = mx.sigmoid(self.w1(x)) * self.w2(x)
        return x 


class GQA(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_qhead == 0
        assert config.n_embd % config.n_kvhead == 0
        assert config.n_qhead % config.n_kvhead == 0

        # KQV for all heads, in one batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_qhead = config.n_qhead
        self.n_kvhead = config.n_kvhead
        self.head_dim = config.n_embd // config.n_qhead
        self.n_rep = self.n_qhead // self.n_kvhead
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flashattention = mx.fast.scaled_dot_product_attention
        self.scale = 1.0 / np.sqrt(self.n_embd // self.n_qhead)
        

    def __call__(self, x, mask=None):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(3, axis=2)
        q = q.reshape(B, T, self.n_qhead, C // self.n_qhead).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kvhead, C // self.n_kvhead).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kvhead, C // self.n_kvhead).transpose(0, 2, 1,3)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if attention_mask is not None:
            y = self.flashattention(q, k, v, mask, scale=self.scale)
        else:
            y = self.flashattention(q, k, v, scale=self.scale)

        y = self.attn_dropout(y)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.swiglu = SwiGLU(4 * config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads                
        self.dim = config.dim
        self.head_dim = self.dim / self.n_heads
        self.attention = GQA(config)
        self.ff = MLP(config)
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn = RMSNorm(config.dim, eps=config.norm_eps)

    def __call__(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        h = h + self.ff(self.ffn(h))
        return h


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.Sequential()
        for layer_id in range(config.n_layers):
            self.layers.append(Block(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def __call__(self, x, start_pos: int):
        _bsz, seqlen = x.shape
        h = self.tok_emb(x)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = mx.full((seqlen, seqlen), float("-inf"))
            mask = mx.triu(mask, diagonal=1)

            mask = hstack([mx.zeros([seqlen, start_pos]), mask)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
