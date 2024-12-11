import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

import os
import sys
import pickle
import time
import random

import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

# what's the mlx equivalent of these?
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

"""
training:
* init from gpt2
implement way later:
    * torch parallel in mlx
    * torch checkpointing
    * figure out the distributed stuff later
    * block cropping
    * logging (wandb)
"""

# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 100 # 1000, 2 for testing
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# # DDP settings
# backend = 'nccl' # 'nccl', 'gloo', etc.
# # system
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = random.sample(range(0, len(data) - block_size), batch_size)
    x = mx.stack([mx.array(data[i: i+block_size], dtype=mx.int64) for i in ix])
    y = mx.stack([mx.array(data[i+1: i+1+block_size], dtype=mx.int64) for i in ix])

    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
"""
new implementation is needed for this ngl 

training from scratch is ok
loading checkpoints - we can load the npz weights, but what about the other values? we need a) checkpoints b) iteration_number c) best validation loss
also initializing the other stuff as well. like the optimizer
"""

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=50304, dropout=dropout) # start with model_args from command line

if init_from == 'scratch':
    print("Initializing a new model from scratch") # init a new model from scratch
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}") # resume training from a checkpoint

    ckpt_weights_path = os.path.join(out_dir, 'ckpt.safetensors')
    ckpt_optimizer_path = os.path.join(out_dir, 'optimizer.safetensors')

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    try: 
        weights, metadata = mx.load(ckpt_weights_path)
        model.load_weights(weights)
        iter_num = metadata['epoch']
        learning_rate = metadata['learning_rate']
        best_val_loss = metadata['best_val_loss']

    except Exception as e:
        print(f"Error occured: {e}. Check if the model arguments are correct for these weights.")

    try:
        state = tree.unflatten(list(mx.load(ckpt_optimizer_path).items()))
    except FileNotFoundError:
        print(f"No optimizer state found at {ckpt_optimizer_path}")
elif init_from.startswith('gpt2'):
    # print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # override_args = dict(dropout=dropout)
    pass # from_pretrained method not implemented yet

if block_size < model.config.block_size:
    pass # block cropping not implemented yet

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))
if init_from == 'resume':
    optimizer.state = state

if compile: # have no idea if this works
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = mx.compile(model)

def loss_fn(model, X, y):
    logits = model(X)
    return nn.losses.cross_entropy(
        logits=logits.reshape(-1, logits.shape[-1]),
        targets=y.reshape(-1),
        reduction='mean'
    ) # implement ignore index later? 

mx.eval(model.parameters())
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# helps estimate an arbitrarily accurate loss over either split using many batches
def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = mx.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            loss, grads = loss_and_grad_fn(model, X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return mx.array(learning_rate * it / warmup_iters)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return mx.array(min_lr)
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return mx.array(min_lr + coeff * (learning_rate - min_lr))


X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model # no ddp, using this as placeholder until i implement ddp
# running_mfu = -1.0
mx.eval(model.parameters())
while True:
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    optimizer.state['learning_rate'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {float(losses['train']):.4f}, val loss {float(losses['val']):.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint: 
            best_val_loss = losses['val']
            if iter_num > 0:
                metadata = {
                    "learning_rate": str(lr),
                    "epoch": str(iter_num),
                    "best_val_loss": str(best_val_loss)
                }
                print(f"saving checkpoints, training metadata, and optimizer to {out_dir}")
                mx.save_safetensors(
                    os.path.join(out_dir, 'ckpt.safetensors'),
                    arrays=model.state_dict(), 
                    metadata=metadata
                )

                optimizer_state = tree_flatten(optimizer.state)
                mx.save_safetensors(
                        os.path.join(out_dir, 'optimizer.safetensors'), 
                        dict(optimizer_state)
                )

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        loss, grads = loss_and_grad_fn(model, X, Y)
        loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        # clip the gradient (not implemented yet)
        # if grad_clip != 0.0
        #     pass
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        # logging (not implemented yet)
        # if iter_num % log_interval == 0:
        #     lossf = loss.item() * gradient_accumulation_steps
        #     if local_iter_num >= 5:
        #         mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #         running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        #     print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break
