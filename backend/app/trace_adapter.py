# app/trace_adapter.py
"""
Trace adapter utilities to instrument a PyTorch model with hooks and
convert tensors into TimeIndexedTensor/simple JSON-serializable structures.

This module is intentionally lightweight: it provides helper functions that
a real training loop can call to convert tensors to the schema used by the API.

Usage pattern:
  - During training, wherever attention/read/write/memory tensors are available,
    call `to_timeindexed(tensor, time=step)` to get a serializable structure.
"""

import numpy as np
from typing import Optional
from .models import TimeIndexedTensor
import torch


def to_timeindexed(tensor: torch.Tensor, time: Optional[int] = None) -> TimeIndexedTensor:
    """
    Convert PyTorch tensor to TimeIndexedTensor with CPU copy and flattened buffer.
    """
    if tensor is None:
        raise ValueError("tensor is None")
    arr = tensor.detach().cpu().numpy()
    shape = list(arr.shape)
    buf = arr.ravel().astype(float).tolist()
    return TimeIndexedTensor(time=time, shape=shape, buffer=buf)


def aggregate_attention_by_head(attn: torch.Tensor) -> torch.Tensor:
    """
    Example helper to transform attention tensor into expected shape [heads, q, k].
    Identity here, kept for semantic clarity.
    """
    return attn
