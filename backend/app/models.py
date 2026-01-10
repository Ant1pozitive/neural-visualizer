# app/models.py
"""
Pydantic/SQLModel models and schemas used by the API.

These Pydantic models mirror the frontend `types.ts` structure so the
frontend and backend have a simple contract.
"""

from typing import List, Optional, Sequence, Dict, Any
from sqlmodel import SQLModel, Field
from pydantic import BaseModel


class Experiment(SQLModel, table=True):
    """
    SQLModel table storing basic experiment metadata.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str
    name: str
    model_type: str
    created_at: str


class TimeIndexedTensor(BaseModel):
    """
    A small, JSON-serializable representation of a time-indexed tensor.
    - shape: list of ints (e.g. [heads, seq_q, seq_k] or [slots, dim])
    - buffer: flattened list of floats (row-major)
    - time: step/time index (optional)
    """
    time: Optional[int] = None
    shape: List[int]
    buffer: List[float]


class ActivationSnapshot(BaseModel):
    """
    Activation snapshot transported to frontend.
    Fields are optional to keep payloads small when some signals are absent.
    """
    step: int
    epoch: Optional[int] = None
    attention: Optional[TimeIndexedTensor] = None
    read_weights: Optional[TimeIndexedTensor] = None
    write_weights: Optional[TimeIndexedTensor] = None
    memory_slots: Optional[TimeIndexedTensor] = None
    reconstructions: Optional[TimeIndexedTensor] = None
    loss: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
