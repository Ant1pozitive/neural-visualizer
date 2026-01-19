# app/storage_core.py
"""
Core Data & Storage API.

- Database models (SQLModel): ExperimentMeta, SnapshotMeta
- High-level functions:
    * init_storage()
    * register_experiment(run_id, name, model_type)
    * persist_activation_snapshot(run_id, activation_snapshot)
    * list_experiments()
    * list_snapshots(run_id, from_step, to_step)
    * load_snapshot(run_id, snapshot_id_or_step, load_tensors=True)
    * cleanup_old_runs(retention_days)
    * export_run(run_id, out_dir)
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from sqlmodel import SQLModel, Field, create_engine, Session, select, Column
from sqlalchemy import JSON as SA_JSON, Integer, String, Text
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

from .models import ActivationSnapshot, TimeIndexedTensor
from .storage_blobs import LocalNPZBackend, S3Backend
from .storage_utils import make_runs_dir

# DB URL (file-based sqlite)
DB_URL = "sqlite:///./storage.db"
engine = create_engine(DB_URL, echo=False, future=True)


# -------------------------------
# Database models
# -------------------------------
class ExperimentMeta(SQLModel, table=True):
    """
    Experiment metadata table.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(sa_column=Column(String, unique=True, index=True))
    name: str
    model_type: str
    created_at: str


class SnapshotMeta(SQLModel, table=True):
    """
    Snapshot metadata: describes a captured snapshot and references blob paths for heavy arrays.
    - blob_paths: JSON dict mapping logical field -> blob relative path or s3 uri
    - meta_json: additional fields (metrics, loss) stored as JSON
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(sa_column=Column(String, index=True))
    step: int = Field(sa_column=Column(Integer, index=True))
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    blob_paths: Dict[str, Any] = Field(default={}, sa_column=Column(SA_JSON))
    meta_json: Dict[str, Any] = Field(default={}, sa_column=Column(SA_JSON))


# -------------------------------
# Storage orchestration / API
# -------------------------------

# instantiate default blob backend (local)
DEFAULT_BLOB_BACKEND = LocalNPZBackend()
# Optionally, env can enable s3 backend; omitted for brevity.

def init_storage():
    """
    Create DB tables and base folders.
    """
    SQLModel.metadata.create_all(engine)
    make_runs_dir("runs")


def register_experiment(run_id: str, name: str, model_type: str) -> ExperimentMeta:
    """
    Insert or return existing experiment metadata row.
    """
    with Session(engine) as sess:
        existing = sess.exec(select(ExperimentMeta).where(ExperimentMeta.run_id == run_id)).first()
        if existing:
            return existing
        exp = ExperimentMeta(run_id=run_id, name=name, model_type=model_type, created_at=datetime.utcnow().isoformat())
        sess.add(exp)
        sess.commit()
        sess.refresh(exp)
        return exp


def _save_tensor_field(run_id: str, field_name: str, tensor_ti: Optional[TimeIndexedTensor], backend=DEFAULT_BLOB_BACKEND) -> Optional[str]:
    """
    Save a TimeIndexedTensor to blob store and return its relative path or URI.
    If tensor_ti is None, return None.
    """
    if tensor_ti is None:
        return None
    # shape and buffer
    shape = tensor_ti.shape
    buffer = tensor_ti.buffer
    # store via backend
    blob_rel = backend.save_buffer(run_id, shape, buffer, meta={"field": field_name})
    return blob_rel


def persist_activation_snapshot(run_id: str, snapshot: ActivationSnapshot, use_backend=None) -> SnapshotMeta:
    """
    Persist ActivationSnapshot: writes any heavy tensors to blob backend, stores metadata row linking to blobs.
    Returns SnapshotMeta row.
    """
    backend = use_backend if use_backend is not None else DEFAULT_BLOB_BACKEND

    blob_paths: Dict[str, Any] = {}

    # Save heavy fields if present
    if snapshot.attention is not None:
        blob_paths["attention"] = _save_tensor_field(run_id, "attention", snapshot.attention, backend=backend)
    if snapshot.read_weights is not None:
        blob_paths["read_weights"] = _save_tensor_field(run_id, "read_weights", snapshot.read_weights, backend=backend)
    if snapshot.write_weights is not None:
        blob_paths["write_weights"] = _save_tensor_field(run_id, "write_weights", snapshot.write_weights, backend=backend)
    if snapshot.memory_slots is not None:
        blob_paths["memory_slots"] = _save_tensor_field(run_id, "memory_slots", snapshot.memory_slots, backend=backend)
    if snapshot.reconstructions is not None:
        blob_paths["reconstructions"] = _save_tensor_field(run_id, "reconstructions", snapshot.reconstructions, backend=backend)

    meta_json = {}
    if snapshot.loss is not None:
        meta_json["loss"] = snapshot.loss
    if snapshot.metrics is not None:
        meta_json["metrics"] = snapshot.metrics

    with Session(engine) as sess:
        row = SnapshotMeta(run_id=run_id, step=int(snapshot.step), created_at=datetime.utcnow().isoformat(), blob_paths=blob_paths, meta_json=meta_json)
        sess.add(row)
        sess.commit()
        sess.refresh(row)
        return row


def list_experiments() -> List[ExperimentMeta]:
    """
    Return all experiments metadata.
    """
    with Session(engine) as sess:
        out = sess.exec(select(ExperimentMeta)).all()
        return out


def list_snapshots(run_id: str, from_step: int = 0, to_step: int = 0, limit: Optional[int] = None) -> List[SnapshotMeta]:
    """
    List snapshot metadata rows for a run and step range.
    """
    with Session(engine) as sess:
        q = select(SnapshotMeta).where(SnapshotMeta.run_id == run_id).where(SnapshotMeta.step >= from_step)
        if to_step:
            q = q.where(SnapshotMeta.step <= to_step)
        q = q.order_by(SnapshotMeta.step)
        if limit:
            q = q.limit(limit)
        rows = sess.exec(q).all()
        return rows


def load_snapshot(run_id: str, snapshot_identifier: Union[int, str], load_tensors: bool = True, backend=DEFAULT_BLOB_BACKEND) -> ActivationSnapshot:
    """
    Load snapshot by id (int) or by step (string 'step:123' or int).
    If load_tensors is False, returns ActivationSnapshot with None tensors but meta fields filled.
    """
    with Session(engine) as sess:
        row = None
        if isinstance(snapshot_identifier, int):
            row = sess.get(SnapshotMeta, snapshot_identifier)
            if row is None:
                raise KeyError("snapshot id not found")
        else:
            # try to interpret as step int
            try:
                step = int(snapshot_identifier)
                row = sess.exec(select(SnapshotMeta).where(SnapshotMeta.run_id == run_id).where(SnapshotMeta.step == step)).first()
            except Exception:
                raise KeyError("invalid snapshot identifier")
        if row is None:
            raise KeyError("snapshot not found")

        # reconstruct ActivationSnapshot
        attn = None
        read_w = None
        write_w = None
        mem = None
        recon = None
        # if load_tensors==True and blob_paths present, load them via backend
        blob_paths = row.blob_paths or {}
        if load_tensors:
            if "attention" in blob_paths and blob_paths["attention"]:
                shape, arr = backend.load_buffer(run_id, blob_paths["attention"])
                attn = TimeIndexedTensor(time=row.step, shape=shape, buffer=arr.ravel().astype(float).tolist())
            if "read_weights" in blob_paths and blob_paths["read_weights"]:
                shape, arr = backend.load_buffer(run_id, blob_paths["read_weights"])
                read_w = TimeIndexedTensor(time=row.step, shape=shape, buffer=arr.ravel().astype(float).tolist())
            if "write_weights" in blob_paths and blob_paths["write_weights"]:
                shape, arr = backend.load_buffer(run_id, blob_paths["write_weights"])
                write_w = TimeIndexedTensor(time=row.step, shape=shape, buffer=arr.ravel().astype(float).tolist())
            if "memory_slots" in blob_paths and blob_paths["memory_slots"]:
                shape, arr = backend.load_buffer(run_id, blob_paths["memory_slots"])
                mem = TimeIndexedTensor(time=row.step, shape=shape, buffer=arr.ravel().astype(float).tolist())
            if "reconstructions" in blob_paths and blob_paths["reconstructions"]:
                shape, arr = backend.load_buffer(run_id, blob_paths["reconstructions"])
                recon = TimeIndexedTensor(time=row.step, shape=shape, buffer=arr.ravel().astype(float).tolist())

        snapshot = ActivationSnapshot(
            step=row.step,
            epoch=None,
            attention=attn,
            read_weights=read_w,
            write_weights=write_w,
            memory_slots=mem,
            reconstructions=recon,
            loss=(row.meta_json.get("loss") if row.meta_json else None),
            metrics=(row.meta_json.get("metrics") if row.meta_json else None)
        )
        return snapshot


def cleanup_old_runs(retention_days: int = 30):
    """
    Delete runs older than retention_days. Deletes DB rows and blob files.
    WARNING: destructive operation.
    """
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    with Session(engine) as sess:
        old = sess.exec(select(ExperimentMeta).where(ExperimentMeta.created_at < cutoff.isoformat())).all()
        for e in old:
            # delete snapshots
            snaps = sess.exec(select(SnapshotMeta).where(SnapshotMeta.run_id == e.run_id)).all()
            for s in snaps:
                # remove blob files locally if local backend
                for field, path in (s.blob_paths or {}).items():
                    if not path:
                        continue
                    # local paths are relative like 'blobs/...' under runs/{run_id}
                    try:
                        run_dir = Path(make_runs_dir()) / e.run_id
                        file_path = run_dir / path
                        if file_path.exists():
                            file_path.unlink()
                        meta_path = str(file_path) + ".meta.json"
                        if Path(meta_path).exists():
                            Path(meta_path).unlink()
                    except Exception:
                        pass
                sess.delete(s)
            # delete experiment row
            sess.delete(e)
        sess.commit()


def export_run(run_id: str, out_dir: str = "export"):
    """
    Export a run into a target directory: copy blobs and create a manifest.json describing snapshots.
    """
    outp = Path(out_dir) / run_id
    outp.mkdir(parents=True, exist_ok=True)
    with Session(engine) as sess:
        snaps = sess.exec(select(SnapshotMeta).where(SnapshotMeta.run_id == run_id).order_by(SnapshotMeta.step)).all()
        manifest = []
        for s in snaps:
            item = {"id": s.id, "step": s.step, "created_at": s.created_at, "meta": s.meta_json, "blobs": {}}
            for field, relpath in (s.blob_paths or {}).items():
                if relpath:
                    # copy file from runs/{run_id}/{relpath} to outp/blobs/
                    src = Path(make_runs_dir()) / run_id / relpath
                    dst_dir = outp / "blobs"
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / Path(relpath).name
                    try:
                        import shutil
                        shutil.copy2(src, dst)
                        item["blobs"][field] = str(Path("blobs") / dst.name)
                        # copy meta sidecar if present
                        mp = str(src) + ".meta.json"
                        if Path(mp).exists():
                            shutil.copy2(mp, str(dst) + ".meta.json")
                    except Exception:
                        # skip
                        item["blobs"][field] = None
            manifest.append(item)
        # write manifest
        with open(outp / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return str(outp)
