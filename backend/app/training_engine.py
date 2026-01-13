# app/training_engine.py
"""
Training & Tracing Engine for instrumentation of PyTorch models.

This module provides:
- TraceCollector: utilities to convert Tensors into API-friendly TimeIndexedTensor
  using the existing trace_adapter.
- Trainer: a lightweight, non-blocking (async) trainer that:
    * runs a training loop over provided dataloader,
    * attempts to request attention/read/write outputs from the model via
      `model(..., return_attn=True)` when supported,
    * converts tensors to ActivationSnapshot objects and registers them
      using experiments.register_snapshot(...)
    * is careful to move data to `device` and to not block the event loop.

The trainer is intentionally generic and model-agnostic. For MemNet specifically,
the model already supports `return_attn=True` and returns `(logits, recon, read_w, write_w)`.
If your model uses different conventions, adjust the "unpack model outputs" section.
"""

from typing import Optional, Callable, Any, Iterable
import asyncio
import time
import traceback

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import experiments
from .trace_adapter import to_timeindexed
from .models import ActivationSnapshot, TimeIndexedTensor

# Type alias for readability
StepCallback = Callable[[ActivationSnapshot], Any]


class TraceCollector:
    """
    Helper class that offers conversion utilities and optionally non-intrusive hooks.
    For many models (including MemNet) it's simplest to call model(..., return_attn=True)
    and unpack returned tensors. TraceCollector centralizes conversions to TimeIndexedTensor.
    """

    @staticmethod
    def tensor_to_timeindexed(tensor: torch.Tensor, time_step: Optional[int] = None) -> TimeIndexedTensor:
        """
        Convert a torch tensor to TimeIndexedTensor using the trace_adapter helper.
        Ensures CPU copy and detaching from graph.
        """
        return to_timeindexed(tensor, time=time_step)

    @staticmethod
    def try_extract_attn_from_model_output(output: Any, time_step: Optional[int] = None):
        """
        Attempt to extract canonical signals from model output.

        Expected output shapes (examples):
         - vanilla output without attn: logits
         - memnet (return_attn=True): (logits, recon, read_weights_hist, write_weights_hist)
        This function tries to be permissive:
          - If output is a tuple/list with len >= 3, it heuristically assigns fields.
          - Otherwise returns None for optional signals.
        """
        attention = None
        recon = None
        read_weights = None
        write_weights = None

        # if model returns tuple-like outputs, try to parse
        if isinstance(output, (tuple, list)):
            # common MemNet pattern: (logits, recon, read_w_hist, write_w_hist)
            if len(output) >= 4:
                # unpack conservatively
                _, recon, read_w_hist, write_w_hist = output[:4]
                # read_w_hist / write_w_hist might be tensors shaped [B, T, ...]
                # choose last timestep (time axis index -1) to represent current step
                try:
                    if isinstance(read_w_hist, torch.Tensor) and read_w_hist.dim() >= 2:
                        # take last time index
                        read_weights = read_w_hist[:, -1, ...]
                    if isinstance(write_w_hist, torch.Tensor) and write_w_hist.dim() >= 2:
                        write_weights = write_w_hist[:, -1, ...]
                except Exception:
                    # fallback: try to convert as-is
                    pass

            # if len == 2, might be (logits, recon)
            elif len(output) == 2:
                _, recon = output
            else:
                # cannot reliably extract
                pass
        else:
            # single tensor or other output
            pass

        # Convert torch tensors to TimeIndexedTensor if present
        def _conv(t):
            try:
                return TraceCollector.tensor_to_timeindexed(t, time_step)
            except Exception:
                return None

        if isinstance(recon, torch.Tensor):
            recon = _conv(recon)
        if isinstance(read_weights, torch.Tensor):
            read_weights = _conv(read_weights)
        if isinstance(write_weights, torch.Tensor):
            write_weights = _conv(write_weights)
        # attention not directly available here; models may return it separately
        return dict(attention=None, reconstructions=recon, read_weights=read_weights, write_weights=write_weights)


class Trainer:
    """
    Trainer runs a training loop for a given PyTorch model and DataLoader and emits
    ActivationSnapshots to the experiments subsystem on every `emit_interval` steps.

    Usage (example):
        trainer = Trainer(model, optimizer, loss_fn, dataloader, device='cpu', run_id=run_id)
        await trainer.start()  # runs until epochs exhausted
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dataloader: DataLoader,
        run_id: str,
        device: str = "cpu",
        num_epochs: int = 1,
        emit_interval: int = 1,
        max_steps: Optional[int] = None,
        grad_clip: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.device = device
        self.run_id = run_id
        self.num_epochs = num_epochs
        self.emit_interval = emit_interval
        self.max_steps = max_steps
        self.grad_clip = grad_clip

        self._stop_requested = False
        self._task: Optional[asyncio.Task] = None
        self.global_step = 0

    def start_in_background(self) -> asyncio.Task:
        """
        Schedule the training loop as an asyncio Task and return it.
        This allows FastAPI startup or other orchestrator to fire-and-forget.
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        return self._task

    def stop(self) -> None:
        """Request graceful stop"""
        self._stop_requested = True
        if self._task and not self._task.done():
            # do not cancel; let the loop check _stop_requested
            pass

    async def _run(self):
        """
        Core training loop. Runs synchronously PyTorch operations but yields control
        to the event loop between steps to avoid blocking.
        """
        try:
            for epoch in range(self.num_epochs):
                if self._stop_requested:
                    break
                for batch in self.dataloader:
                    if self._stop_requested:
                        break
                    self.global_step += 1
                    # -- prepare batch --
                    # Expected batch shapes: (inputs, targets) or single tensor input
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                    else:
                        inputs = batch.to(self.device)
                        # create dummy targets if none provided (e.g., reconstruction)
                        targets = None

                    # -- forward (try to request attention outputs if supported) --
                    # Many models accept `return_attn=True` to emit read/write weights.
                    # We call the model inside torch.no_grad=False (we need gradients).
                    try:
                        # Try a call that requests attn; if model doesn't accept, fallback.
                        try:
                            output = self.model(inputs, return_attn=True)
                        except TypeError:
                            # model doesn't accept return_attn flag
                            output = self.model(inputs)
                    except Exception as e:
                        # Forward failed: log and stop training
                        print(f"[Trainer] Model forward error at step {self.global_step}: {e}")
                        traceback.print_exc()
                        self._stop_requested = True
                        break

                    # Unpack logits if returned in tuple or treat single output as logits
                    logits = None
                    recon = None
                    read_w_ti = None
                    write_w_ti = None

                    if isinstance(output, (tuple, list)):
                        # If model returned attention snapshots (MemNet), our heuristics will parse them
                        # Keep original output for loss calculation attempts
                        # Attempt to get logits = first element
                        if len(output) > 0 and isinstance(output[0], torch.Tensor):
                            logits = output[0]
                        # Try to extract attn & recon using TraceCollector
                        ext = TraceCollector.try_extract_attn_from_model_output(output, time_step=self.global_step)
                        recon = ext.get("reconstructions")
                        read_w_ti = ext.get("read_weights")
                        write_w_ti = ext.get("write_weights")
                    elif isinstance(output, torch.Tensor):
                        logits = output
                    else:
                        # unknown output type — skip
                        pass

                    # -- compute loss if possible --
                    loss_val = None
                    if logits is not None and targets is not None:
                        # try to align shapes for a typical classification/regression loss
                        try:
                            loss = self.loss_fn(logits, targets)
                            loss_val = float(loss.detach().cpu().item())
                        except Exception as e:
                            # can't compute loss with given outputs/targets
                            loss = None
                            loss_val = None
                    else:
                        loss = None

                    # -- backward + optimizer step --
                    if loss is not None:
                        self.optimizer.zero_grad()
                        loss.backward()
                        if self.grad_clip:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                    # -- try to capture model-level memory or other tensors --
                    memory_slots_ti = None
                    # If model has attribute `memory` exposing slots as a tensor, attempt to capture it
                    try:
                        mem = getattr(self.model, "memory", None)
                        if mem is not None:
                            # many memory modules hold slots in an attribute like `slots` or similar.
                            # We try common attribute names; if none present, skip.
                            if hasattr(mem, "slots"):
                                slots_tensor = getattr(mem, "slots")
                                if isinstance(slots_tensor, torch.Tensor):
                                    memory_slots_ti = TraceCollector.tensor_to_timeindexed(slots_tensor, time_step=self.global_step)
                            elif hasattr(mem, "get_memory_state"):
                                st = mem.get_memory_state()
                                if isinstance(st, torch.Tensor):
                                    memory_slots_ti = TraceCollector.tensor_to_timeindexed(st, time_step=self.global_step)
                    except Exception:
                        # Non-critical: memory introspection failed — continue
                        pass

                    # -- prepare ActivationSnapshot --
                    snapshot = ActivationSnapshot(
                        step=self.global_step,
                        epoch=epoch,
                        attention=None,  # optional: models may provide attention separately
                        read_weights=read_w_ti,
                        write_weights=write_w_ti,
                        memory_slots=memory_slots_ti,
                        reconstructions=recon,
                        loss=loss_val,
                        metrics=None
                    )

                    # register snapshot (persist + notify subscribers)
                    try:
                        experiments.register_snapshot(self.run_id, snapshot)
                    except Exception as e:
                        print(f"[Trainer] Failed to register snapshot: {e}")

                    # yield to event loop regularly to avoid blocking
                    if self.global_step % 4 == 0:
                        await asyncio.sleep(0)

                    # check step limit
                    if self.max_steps and self.global_step >= self.max_steps:
                        self._stop_requested = True
                        break

                # epoch end
            # training finished
        except asyncio.CancelledError:
            print("[Trainer] Cancelled")
        except Exception as e:
            print(f"[Trainer] Unexpected exception: {e}")
            traceback.print_exc()
        finally:
            print(f"[Trainer] Training ended for run_id={self.run_id}, steps={self.global_step}")
