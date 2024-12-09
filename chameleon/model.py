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
    - GQA
    - RoPE
    - self attention using RoPE
    - FFN with swiglu activation
    - residual connections
- model (vqgan? the image one)
- model configs (vqgan stays the same)
- training code
    - training scheduling as per the recipe in chameleon, then c3mleon
- data preparation
"""
