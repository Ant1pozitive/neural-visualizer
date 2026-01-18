# app/instrumentation/utils.py
"""
Utility helpers for instrumentation.
"""

from typing import Optional, Tuple
import torch


def extract_tensor_name(tensor) -> str:
    """
    Try to extract a human-friendly name for a tensor or module output.
    This is heuristic-based; used for labeling captured activations.
    """
    if hasattr(tensor, "grad_fn") and tensor.grad_fn is not None:
        return tensor.grad_fn.__class__.__name__
    if isinstance(tensor, torch.nn.Parameter):
        return "parameter"
    return type(tensor).__name__
