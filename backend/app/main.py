# app/main.py
"""
Application entrypoint: assemble FastAPI app, include router and startup tasks.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import storage
from .api import router as api_router
from .experiments import start_mock_run
import asyncio

app = FastAPI(title="NN Visualization API")

# Allow CORS from local dev frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    """
    Initialize DB and create one demo run automatically for convenience.
    """
    storage.init_db()
    # start a demo run in background so frontend can connect immediately
    # Note: start_mock_run schedules async tasks internally
    run_id = start_mock_run(name="auto-demo", model_type="memnet")
    print(f"Auto demo run started: {run_id}")


# include API router
app.include_router(api_router, prefix="", tags=["api"])
