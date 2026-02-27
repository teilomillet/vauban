"""MLX tensor operations backend.

Exports exactly the symbols listed in ``_ops_contract.OPS_CONTRACT``.
All re-exports — zero overhead, no wrappers.
"""

import mlx.core as _mx
from mlx.utils import tree_flatten as tree_flatten  # explicit re-export

# Array creation
array = _mx.array
zeros = _mx.zeros
zeros_like = _mx.zeros_like
ones = _mx.ones
arange = _mx.arange
full = _mx.full

# Reductions
sum = _mx.sum
mean = _mx.mean

# Element-wise math
abs = _mx.abs
exp = _mx.exp
log = _mx.log
sqrt = _mx.sqrt
maximum = _mx.maximum
minimum = _mx.minimum
outer = _mx.outer
arccos = _mx.arccos
cos = _mx.cos
clip = _mx.clip
where = _mx.where

# Selection / sorting
argmax = _mx.argmax
argpartition = _mx.argpartition
sort = _mx.sort
argsort = _mx.argsort
softmax = _mx.softmax
matmul = _mx.matmul

# Manipulation
concatenate = _mx.concatenate
stack = _mx.stack
reshape = _mx.reshape
expand_dims = _mx.expand_dims

# I/O
load = _mx.load
save_safetensors = _mx.save_safetensors

# Gradient
value_and_grad = _mx.value_and_grad
stop_gradient = _mx.stop_gradient

# Evaluation
eval = _mx.eval

# Types / dtypes
float32 = _mx.float32
float16 = _mx.float16
bfloat16 = _mx.bfloat16
int32 = _mx.int32
uint32 = _mx.uint32
bool_ = _mx.bool_

# Sub-namespaces
linalg = _mx.linalg
random = _mx.random

# Stream
cpu = _mx.cpu
