# app/storage_utils.py
"""
Utility functions for Data & Storage layer.
"""

import os
import uuid
from pathlib import Path
from typing import Tuple


def make_runs_dir(base: str = "runs") -> str:
    """
    Ensure runs base directory exists and return its path.
    """
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def make_run_dir(run_id: str, base: str = "runs") -> str:
    """
    Ensure directory for specific run exists.
    """
    base_dir = make_runs_dir(base)
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def unique_blob_name(prefix: str = "blob", ext: str = ".npz") -> str:
    """
    Generate a collision-free filename for a blob.
    """
    return f"{prefix}_{uuid.uuid4().hex}{ext}"


def safe_join(base_dir: str, filename: str) -> str:
    """
    Safely join path elements and prevent path traversal.
    """
    base = Path(base_dir).resolve()
    candidate = (base / filename).resolve()
    if not str(candidate).startswith(str(base)):
        raise ValueError("Unsafe filename/path")
    return str(candidate)
