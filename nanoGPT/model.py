import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim

import inspect
import math
import numpy as np
from dataclasses import dataclass

"""
implement:
* blocks:
    * layernorm
    * causal self-attention
    * transformer 
    * MLP
    * transformer block
    * GPT
        * model setup
        * skip weight init
        * __call__ prop
        * crop block size
        * init from pretrained
        * configure optimizer
        * estimate mfu?
        * autoregressive generate
* setup
    * config

goals:
* implement this by myself, no claude assist or copying other people's code
* train the model, get the loss curves and a model i can use to generate text
* use einsums as much as possible
* train a chameleon model based on this architecture
"""

class LayerNorm(nn.Module):

    def __init__(self, ndim: int, bias: bool=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(dims=ndim, eps=1e-5, affine=True, bias=bias)

    def __call__(self, x):
        return self.layernorm(x)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # embeddings must be distributed evenly amongst heads, integer values
        # KQV for all heads, in one batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flashattention = mx.fast.scaled_dot_product_attention
        self.scale = 1.0 / np.sqrt(self.n_embd)
        
    def __call__(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        q, k, v = self.c_attn(x).split(3, axis=2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1,3)
        # k = rearrange(k, 'b t (h c2) -> b h t c2', h=self.n_head)
        # q = rearrange(q, 'b t (h c2) -> b h t c2', h=self.n_head)
        # v = rearrange(v, 'b t (h c2) -> b h t c2', h=self.n_head)

        y = self.flashattention(q, k, v, scale=self.scale)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        # y = rearrange(y, 'b h t c2 -> b t (h c2)')

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def __call__(self, idx, train=True):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = mx.arange(0, t, dtype=mx.int64)
        
        # forward the GPT model itself
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        if train:
            logits = self.lm_head(x)
        else:
            logits = self.lm_head(x[:, [-1], :])
        
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optimizer = optim.AdamW( # specify eps later
            learning_rate=learning_rate, 
            betas=betas, 
            weight_decay=weight_decay
        ) 
        return optimizer

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, train=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = mx.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = mx.softmax(logits, axis=1)
            idx_next = mx.random.categorical(probs, num_samples=1)
            idx = mx.concatenate([idx, idx_next], axis=1) 

        return idx
