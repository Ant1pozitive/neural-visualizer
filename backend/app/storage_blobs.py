# app/storage_blobs.py
"""
Blob storage backends for saving/loading tensor buffers.

- LocalNPZBackend: saves TimeIndexedTensor-like buffers to compressed .npz files.
- S3Backend (optional): uploads blobs to S3 (requires boto3 and config in env).
"""

from typing import Dict, Any, Optional
import numpy as np
import os
from pathlib import Path
from .storage_utils import make_run_dir, safe_join, unique_blob_name
import json

# lazy import for S3 to keep optional dependency
try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
    _HAS_BOTO3 = True
except Exception:
    _HAS_BOTO3 = False


class LocalNPZBackend:
    """
    Simple backend that stores blobs as compressed NumPy .npz files under runs/{run_id}/blobs/.
    File format: a single array named 'buffer' + JSON metadata file with 'shape' and optional attrs.
    """

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = base_dir
        self.blobs_dirname = "blobs"

    def _ensure_blobs_dir(self, run_id: str) -> str:
        run_dir = make_run_dir(run_id, base=self.base_dir)
        blobs_dir = Path(run_dir) / self.blobs_dirname
        blobs_dir.mkdir(parents=True, exist_ok=True)
        return str(blobs_dir)

    def save_buffer(self, run_id: str, shape: list, buffer: list, meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Save provided buffer (flattened list) and shape as compressed npz.
        Returns relative path (run-local) to saved blob.
        """
        blobs_dir = self._ensure_blobs_dir(run_id)
        fname = unique_blob_name("tensor")
        file_path = Path(blobs_dir) / fname
        # convert to numpy array (float64)
        arr = np.asarray(buffer, dtype=np.float64)
        # store as single array 'buffer' and include metadata in JSON sidecar
        np.savez_compressed(str(file_path), buffer=arr)
        # write metadata next to file for shape and optional meta
        meta_path = str(file_path) + ".meta.json"
        meta_content = {"shape": shape}
        if meta:
            meta_content.update(meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            json_str = json.dumps(meta_content)
            f.write(json_str)
        # return path relative to run dir (e.g., blobs/...)
        run_dir = Path(make_run_dir(run_id, base=self.base_dir))
        return str(Path("blobs") / fname)

    def load_buffer(self, run_id: str, rel_path: str):
        """
        Load buffer from previously saved blob.
        Returns tuple (shape, numpy_array)
        """
        run_dir = Path(make_run_dir(run_id, base=self.base_dir))
        full_path = safe_join(str(run_dir), rel_path)
        # full_path points to .npz file
        if not Path(full_path).exists():
            raise FileNotFoundError(full_path)
        npz = np.load(full_path)
        arr = npz["buffer"]
        meta_path = full_path + ".meta.json"
        shape = None
        if Path(meta_path).exists():
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                shape = meta.get("shape")
        return shape, arr


class S3Backend:
    """
    Optional S3 backend for blob storage. Requires boto3.
    This implementation stores .npz bytes in S3 under a given bucket prefix and keeps metadata as object metadata or sidecar.
    """

    def __init__(self, bucket: str, prefix: str = "runs", region_name: Optional[str] = None):
        if not _HAS_BOTO3:
            raise RuntimeError("boto3 not installed; cannot use S3Backend")
        self.s3 = boto3.client("s3", region_name=region_name)
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

    def save_buffer(self, run_id: str, shape: list, buffer: list, meta: Optional[Dict[str, Any]] = None) -> str:
        import io, json
        arr = np.asarray(buffer, dtype=np.float64)
        mem = io.BytesIO()
        # save compressed npz into memory
        np.savez_compressed(mem, buffer=arr)
        mem.seek(0)
        key = f"{self.prefix}/{run_id}/blobs/{unique_blob_name('tensor')}"
        try:
            # upload
            self.s3.upload_fileobj(mem, self.bucket, key)
            # optional: upload metadata as sidecar
            if meta is None:
                meta = {}
            meta_obj = {"shape": shape}
            meta_obj.update(meta)
            meta_key = key + ".meta.json"
            self.s3.put_object(Bucket=self.bucket, Key=meta_key, Body=json.dumps(meta_obj).encode("utf-8"))
        except (BotoCoreError, ClientError) as e:
            raise
        # return key (s3 uri-like)
        return f"s3://{self.bucket}/{key}"

    def load_buffer(self, run_id: str, s3_key: str):
        import io, json
        # s3_key expected like s3://bucket/prefix/... or key relative to bucket
        if s3_key.startswith("s3://"):
            # parse
            _, _, rest = s3_key.partition("s3://")
            bucket, _, key = rest.partition("/")
            bucket = bucket
            key = key
        else:
            # assume key already relative to configured bucket
            bucket = self.bucket
            key = s3_key
        mem = io.BytesIO()
        self.s3.download_fileobj(bucket, key, mem)
        mem.seek(0)
        npz = np.load(mem)
        arr = npz["buffer"]
        # retrieve sidecar metadata
        meta_key = key + ".meta.json"
        meta_shape = None
        try:
            meta_obj = self.s3.get_object(Bucket=bucket, Key=meta_key)
            meta_str = meta_obj["Body"].read().decode("utf-8")
            meta = json.loads(meta_str)
            meta_shape = meta.get("shape")
        except Exception:
            meta_shape = None
        return meta_shape, arr
