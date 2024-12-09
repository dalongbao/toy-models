import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optim as optim
from mlx.utils import tree_flatten, tree_unflatten

import argparse
import glob
import json
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from model import LlamaConfig, Llama

out_dir = 'out'
eval_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_kvhead = None # hollup
n_qhead = None # hollup
dropout = 0.0
bias = False

vocab_size = 
norm_eps = 
