# app/instrumentation/instrumentor.py
"""
Instrumentor: attach/detach hooks to PyTorch models and emit ActivationSnapshots.

API:
    inst = Instrumentor(model, run_id, signals=["attention","activations","memory","gradients"])
    inst.attach()
    # training runs...
    inst.detach()

Design:
- Use forward hooks to capture outputs/activations.
- Use backward hooks or register_full_backward_hook for gradient capture where needed.
- Use adapters (HuggingFaceAdapter or GenericTorchAdapter) to parse model outputs.
- Convert tensors to TimeIndexedTensor via trace_adapter.to_timeindexed.
- Offload heavy storage/persistence to background thread using asyncio.to_thread to avoid blocking training loop.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import weakref
import inspect
import traceback

import torch
from torch import nn

from ..models import ActivationSnapshot, TimeIndexedTensor
from .. import experiments
from ..trace_adapter import to_timeindexed
from .adapters import HuggingFaceAdapter, GenericTorchAdapter


class Instrumentor:
    def __init__(self, model: nn.Module, run_id: str, signals: Optional[List[str]] = None, time_step_fn: Optional[Callable[[], int]] = None):
        """
        Initialize instrumentor.

        Args:
            model: PyTorch model to instrument.
            run_id: experiment run id to which snapshots will be registered.
            signals: list of signals to capture: "attention", "activations", "gradients", "memory"
            time_step_fn: optional callable returning current logical step; if None, Instrumentor will increment an internal counter.
        """
        self.model = model
        self.run_id = run_id
        self.signals = set(signals or ["attention", "activations", "memory", "gradients"])
        self.time_step_fn = time_step_fn
        self._hooks: List[Tuple[Any, Callable]] = []
        self._back_hooks: List[Tuple[Any, Callable]] = []
        self._attached = False
        self._step = 0
        # choose adapter: prefer HuggingFace if available & model appears to be HF
        self.hf_adapter = HuggingFaceAdapter() if True else None
        self.generic_adapter = GenericTorchAdapter()
        # keep weakref to avoid circular refs
        self._model_ref = weakref.ref(model)

    def _next_step(self) -> int:
        if self.time_step_fn is not None:
            return int(self.time_step_fn())
        self._step += 1
        return self._step

    def attach(self):
        """
        Attach forward hooks and backward hooks to the model.
        Forward hook is placed at top-level module to intercept outputs,
        and also at child modules if activations capture is requested.
        """
        if self._attached:
            return

        # top-level forward hook to inspect outputs
        def _forward_hook(module, inputs, outputs):
            """
            This hook is executed in the same thread as forward.
            We schedule heavy processing to background via asyncio.to_thread.
            """
            try:
                step = self._next_step()
                # schedule processing asynchronously (non-blocking)
                asyncio.get_event_loop().call_soon_threadsafe(asyncio.create_task, self._process_forward(outputs, step))
            except Exception:
                # fallback: try synchronous but safe
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self._process_forward(outputs, self._next_step()))
                    loop.close()
                except Exception:
                    traceback.print_exc()

        # register on top module
        hook_handle = self.model.register_forward_hook(_forward_hook)
        self._hooks.append((self.model, hook_handle))

        # attach activation hooks to child modules if requested
        if "activations" in self.signals:
            # We'll attach forward hooks to leaf modules (no children) to capture activation tensors.
            for name, module in self.model.named_modules():
                # skip the top-level module since we already have it
                if module is self.model:
                    continue
                # attach only to modules that are computational (e.g., Linear, Conv, Attention). Use heuristic:
                if len(list(module.children())) == 0:
                    # forward hook capturing outputs
                    def make_activation_hook(mod_name):
                        def _act_hook(mod, inp, outp):
                            try:
                                step = self._next_step()
                                asyncio.get_event_loop().call_soon_threadsafe(asyncio.create_task, self._process_activation(mod_name, outp, step))
                            except Exception:
                                # ignore activation hook failure
                                pass
                        return _act_hook
                    h = module.register_forward_hook(make_activation_hook(name))
                    self._hooks.append((module, h))

        # gradients: register full backward hook on top-level module to capture gradients at end
        if "gradients" in self.signals:
            # PyTorch 2.0: register_full_backward_hook is available. Use try/except for compatibility.
            try:
                def _back_hook(module, grad_input, grad_output):
                    # grad_input/grad_output are tuples of tensors/None
                    try:
                        step = self._next_step()
                        asyncio.get_event_loop().call_soon_threadsafe(asyncio.create_task, self._process_gradients(grad_input, grad_output, step))
                    except Exception:
                        pass
                bh = None
                if hasattr(self.model, "register_full_backward_hook"):
                    bh = self.model.register_full_backward_hook(lambda m, gi, go: _back_hook(m, gi, go))
                else:
                    # fallback: register backward hook on parameters (costly)
                    for p in self.model.parameters():
                        if p.requires_grad:
                            bh = p.register_hook(lambda grad, p_ref=p: asyncio.get_event_loop().call_soon_threadsafe(asyncio.create_task, self._process_param_grad(p_ref, grad, self._next_step())))
                            self._back_hooks.append((p, bh))
                    bh = None
                if bh is not None:
                    self._back_hooks.append((self.model, bh))
            except Exception:
                # degrade gracefully
                pass

        self._attached = True

    def detach(self):
        """
        Remove all registered hooks to restore model to original state.
        """
        for mod, handle in self._hooks:
            try:
                handle.remove()
            except Exception:
                # some handles may not support remove; ignore
                pass
        self._hooks.clear()

        for obj, handle in self._back_hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self._back_hooks.clear()
        self._attached = False

    async def _process_forward(self, outputs: Any, step: int):
        """
        Process outputs captured by forward hook. Use adapters to extract signals, build ActivationSnapshot,
        and register snapshot via experiments.register_snapshot in a thread to avoid blocking.
        """
        try:
            model = self._model_ref()
            # choose adapter heuristically
            adapter = None
            if self.hf_adapter is not None:
                # If model seems like HF model (has config or transformers spec), attempt HF adapter first
                try:
                    # if outputs has attribute 'attentions' and model has attribute 'config' -> likely HF
                    if hasattr(outputs, "attentions") or hasattr(model, "config"):
                        adapter = self.hf_adapter
                except Exception:
                    adapter = None
            if adapter is None:
                adapter = self.generic_adapter

            # extract signals via adapter
            signals = adapter.extract_signals(outputs, model, time_step=step)

            # Also, if "activations" signal is requested but adapter didn't provide, try to convert outputs directly
            recon_ti = signals.get("reconstructions")
            read_w_ti = signals.get("read_weights")
            write_w_ti = signals.get("write_weights")
            attention_ti = signals.get("attention")
            memory_ti = signals.get("memory_slots")

            # Merge into ActivationSnapshot
            snapshot = ActivationSnapshot(
                step=step,
                epoch=None,
                attention=attention_ti,
                read_weights=read_w_ti,
                write_weights=write_w_ti,
                memory_slots=memory_ti,
                reconstructions=recon_ti,
                loss=None,
                metrics=None
            )

            # Persist/notify using thread to avoid blocking
            await asyncio.to_thread(experiments.register_snapshot, self.run_id, snapshot)
        except Exception:
            traceback.print_exc()

    async def _process_activation(self, module_name: str, activation: Any, step: int):
        """
        Called for leaf-module activations when 'activations' signal requested.
        Convert activation to TimeIndexedTensor and register as lightweight snapshot.
        """
        try:
            # activation may be tensor or tuple/list/dict containing tensors
            tensor = None
            if isinstance(activation, torch.Tensor):
                tensor = activation.detach()
            elif isinstance(activation, (list, tuple)):
                # choose first tensor-like
                for item in activation:
                    if isinstance(item, torch.Tensor):
                        tensor = item.detach()
                        break
            elif isinstance(activation, dict):
                for v in activation.values():
                    if isinstance(v, torch.Tensor):
                        tensor = v.detach()
                        break
            if tensor is None:
                return
            ti = to_timeindexed(tensor, time=step)
            # place activation into memory_slots for frontend convenience (lightweight)
            snapshot = ActivationSnapshot(step=step, epoch=None, attention=None, read_weights=None, write_weights=None, memory_slots=ti, reconstructions=None, loss=None, metrics={"activation_module": 0.0})
            await asyncio.to_thread(experiments.register_snapshot, self.run_id, snapshot)
        except Exception:
            traceback.print_exc()

    async def _process_gradients(self, grad_input, grad_output, step: int):
        """
        Capture gradients from module-level backward hook. This is kept lightweight:
        compute simple statistics (norms) and register as metric snapshot.
        """
        try:
            # flatten grad tensors and compute L2 norm as a metric
            total_norm = 0.0
            count = 0
            for g in list(grad_input) + list(grad_output):
                try:
                    if isinstance(g, torch.Tensor):
                        total_norm += float(torch.linalg.norm(g).item() ** 2)
                        count += 1
                except Exception:
                    continue
            if count > 0:
                l2 = (total_norm ** 0.5)
            else:
                l2 = 0.0
            snapshot = ActivationSnapshot(step=step, epoch=None, attention=None, read_weights=None, write_weights=None, memory_slots=None, reconstructions=None, loss=None, metrics={"grad_l2": float(l2)})
            await asyncio.to_thread(experiments.register_snapshot, self.run_id, snapshot)
        except Exception:
            traceback.print_exc()

    async def _process_param_grad(self, param, grad, step: int):
        """
        Called for parameter-level hooks. Register norm per-param as metric.
        """
        try:
            n = float(torch.linalg.norm(grad).item()) if grad is not None else 0.0
            snapshot = ActivationSnapshot(step=step, epoch=None, attention=None, read_weights=None, write_weights=None, memory_slots=None, reconstructions=None, loss=None, metrics={f"param_grad_{id(param)}": n})
            await asyncio.to_thread(experiments.register_snapshot, self.run_id, snapshot)
        except Exception:
            traceback.print_exc()
