# app/instrumentation/adapters.py
"""
Adapters for extracting signals from models.

- HuggingFaceAdapter: if `transformers` is present, uses `return_attentions=True` and
  extracts `.attentions` from model outputs (a tuple/list or BaseModelOutput).
- GenericTorchAdapter: for arbitrary models, it provides heuristics:
  - If model returns tuple/list, inspect elements for tensors named/looking like attention or memory.
  - If model has named submodules/attributes like 'memory', 'memory_bank', 'attention', use them.
Adapters expose a consistent method `extract_signals(model_output, model)` returning a dict:
{
  "attention": TimeIndexedTensor | None,
  "read_weights": TimeIndexedTensor | None,
  "write_weights": TimeIndexedTensor | None,
  "memory_slots": TimeIndexedTensor | None,
  "reconstructions": TimeIndexedTensor | None,
  ...
}
"""

from typing import Any, Dict, Optional
import importlib

from .utils import extract_tensor_name
from ..trace_adapter import to_timeindexed
import torch


class HuggingFaceAdapter:
    """
    Adapter for HuggingFace Transformers.

    Usage:
        out = model(inputs, output_attentions=True, return_dict=True)
        adapter = HuggingFaceAdapter()
        signals = adapter.extract_signals(out, model, time_step)
    """

    def __init__(self):
        # try to import types - optional
        self.transformers = importlib.util.find_spec("transformers") is not None

    def extract_signals(self, model_output: Any, model: torch.nn.Module, time_step: Optional[int] = None) -> Dict[str, Optional[object]]:
        """
        Convert huggingface outputs to our TimeIndexedTensor-compatible dict.
        """
        out = {
            "attention": None,
            "read_weights": None,
            "write_weights": None,
            "memory_slots": None,
            "reconstructions": None
        }

        # HuggingFace BaseModelOutput usually has .attentions (tuple of tensors per layer)
        try:
            attentions = getattr(model_output, "attentions", None)
            if attentions is not None:
                # attentions is tuple(layer tensors), each has shape [batch, heads, seq_q, seq_k]
                # We'll stack/aggregate into single tensor: [heads, seq_q, seq_k] by averaging across batch & layers
                # For frontend we can average across layers or provide first layer — here we average across layers & batch
                # Convert to one numpy-like structure via trace_adapter when possible.
                # For simplicity, convert first layer's attention (common case)
                if len(attentions) > 0:
                    # attentions[0] is tensor [B, H, Q, K]
                    t = attentions[0]
                    # average batch dimension -> [H, Q, K]
                    t_avg = t.mean(dim=0)
                    out["attention"] = to_timeindexed(t_avg.detach(), time_step)
        except Exception:
            pass

        # attempt memory slot extraction if model has attribute 'memory' or 'mem'
        try:
            mem = getattr(model, "memory", None) or getattr(model, "mem", None)
            if mem is not None:
                # try common attribute names
                if hasattr(mem, "slots"):
                    slots = getattr(mem, "slots")
                    if isinstance(slots, torch.Tensor):
                        out["memory_slots"] = to_timeindexed(slots.detach(), time_step)
                elif hasattr(mem, "get_memory_state"):
                    st = mem.get_memory_state()
                    if isinstance(st, torch.Tensor):
                        out["memory_slots"] = to_timeindexed(st.detach(), time_step)
        except Exception:
            pass

        return out


class GenericTorchAdapter:
    """
    Generic heuristics-based adapter for arbitrary torch modules.
    This adapter inspects:
      - model output tuple/list elements for likely signals (by size, name or semantics)
      - model attributes (memory, memory_bank, attention)
      - named modules (Attention, MultiheadAttention)
    """

    def __init__(self):
        # heuristics parameters (tunable)
        self.min_attention_rank = 3  # expecting [heads, q, k] or [batch, heads, q, k]
        self.max_heads = 64

    def _looks_like_attention(self, tensor: torch.Tensor) -> bool:
        """
        Heuristic: 3D/4D tensor with one dimension <= max_heads looking like attention.
        """
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.dim() == 3:
            # [heads, q, k]
            h = tensor.shape[0]
            return 1 <= h <= self.max_heads
        if tensor.dim() == 4:
            # [batch, heads, q, k]
            h = tensor.shape[1]
            return 1 <= h <= self.max_heads
        return False

    def extract_signals(self, model_output: Any, model: torch.nn.Module, time_step: Optional[int] = None) -> Dict[str, Optional[object]]:
        out = {
            "attention": None,
            "read_weights": None,
            "write_weights": None,
            "memory_slots": None,
            "reconstructions": None
        }

        # Inspect output
        if isinstance(model_output, (list, tuple)):
            for item in model_output:
                try:
                    if isinstance(item, torch.Tensor):
                        if self._looks_like_attention(item):
                            # normalize to [heads, q, k] by averaging batch if present
                            t = item
                            if t.dim() == 4:
                                t = t.mean(dim=0)
                            out["attention"] = to_timeindexed(t.detach(), time_step)
                            continue
                        # heuristics for read/write weights: 2D shapes with small second dim (slots)
                        if item.dim() == 2:
                            # if second dim small -> possibly [batch, slots] or [heads, slots] etc.
                            if item.shape[1] <= 256:
                                # choose as read/write candidate if not set
                                if out["read_weights"] is None:
                                    out["read_weights"] = to_timeindexed(item.detach(), time_step)
                                    continue
                                elif out["write_weights"] is None:
                                    out["write_weights"] = to_timeindexed(item.detach(), time_step)
                                    continue
                        # reconstruction: shape [batch, seq, dim] or [seq, dim]
                        if item.dim() in (2, 3) and (item.dim() == 2 or item.dim() == 3):
                            # prefer to set recon if shape fits
                            if out["reconstructions"] is None:
                                out["reconstructions"] = to_timeindexed(item.detach(), time_step)
                                continue
                except Exception:
                    pass

        # Inspect model attributes for memory-like modules
        try:
            mem = getattr(model, "memory", None) or getattr(model, "mem", None)
            if mem is not None:
                if hasattr(mem, "slots"):
                    slots = getattr(mem, "slots")
                    if isinstance(slots, torch.Tensor):
                        out["memory_slots"] = to_timeindexed(slots.detach(), time_step)
                elif hasattr(mem, "get_memory_state"):
                    st = mem.get_memory_state()
                    if isinstance(st, torch.Tensor):
                        out["memory_slots"] = to_timeindexed(st.detach(), time_step)
        except Exception:
            pass

        # If no attention found but model contains MultiheadAttention modules, try to read their attn weights
        if out["attention"] is None:
            for name, module in model.named_modules():
                # PyTorch built-in MHA does not expose attention weights by default.
                # Some custom modules might keep last_attn attribute — try heuristics.
                if hasattr(module, "last_attn") and isinstance(getattr(module, "last_attn"), torch.Tensor):
                    try:
                        t = getattr(module, "last_attn")
                        if t.dim() == 4:
                            t_avg = t.mean(dim=0)
                            out["attention"] = to_timeindexed(t_avg.detach(), time_step)
                            break
                    except Exception:
                        pass

        return out
