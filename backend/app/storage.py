# app/storage.py
"""
Storage utilities:
- metadata persisted in SQLite via SQLModel
- snapshots persisted as JSONL files under runs/{run_id}/snapshots.jsonl

This approach keeps implementation simple and portable.
For production, swap JSONL -> Parquet/NPZ and SQLModel -> real RDBMS.
"""

import os
import json
from typing import Iterable, List
from sqlmodel import SQLModel, Session, create_engine, select
from .models import Experiment, ActivationSnapshot
from datetime import datetime
from pathlib import Path

DB_URL = "sqlite:///./backend.db"
engine = create_engine(DB_URL, echo=False)


def init_db():
    """
    Initialize DB and tables.
    """
    SQLModel.metadata.create_all(engine)


def register_experiment(run_id: str, name: str, model_type: str) -> Experiment:
    """
    Create metadata entry for a new experiment.
    """
    created_at = datetime.utcnow().isoformat()
    exp = Experiment(run_id=run_id, name=name, model_type=model_type, created_at=created_at)
    with Session(engine) as sess:
        sess.add(exp)
        sess.commit()
        sess.refresh(exp)
    return exp


def list_experiments() -> List[Experiment]:
    with Session(engine) as sess:
        q = select(Experiment)
        results = sess.exec(q).all()
    return results


def snapshots_dir(run_id: str) -> str:
    base = Path("runs") / str(run_id)
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def append_snapshot(run_id: str, snapshot: ActivationSnapshot):
    """
    Append a JSON-serialized snapshot to runs/{run_id}/snapshots.jsonl.
    """
    d = snapshot.dict()
    path = Path(snapshots_dir(run_id)) / "snapshots.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d) + "\n")


def load_snapshots(run_id: str, from_step: int = 0, to_step: int = 0) -> List[ActivationSnapshot]:
    """
    Load snapshots for run_id. If to_step==0, return all from from_step.
    """
    path = Path(snapshots_dir(run_id)) / "snapshots.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            step = data.get("step", 0)
            if step < from_step:
                continue
            if to_step and step > to_step:
                continue
            out.append(ActivationSnapshot(**data))
    return out
