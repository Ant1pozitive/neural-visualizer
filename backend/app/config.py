# app/config.py
"""
Centralized configuration for API & orchestration layer.
"""

from pydantic import BaseModel


class Settings(BaseModel):
    snapshots_dir: str = "runs"
    max_in_memory_snapshots: int = 2000
    enable_autodemo: bool = True


settings = Settings()
