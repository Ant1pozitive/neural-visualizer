# app/api.py
"""
FastAPI router definitions and WebSocket manager.

Endpoints:
- GET  /experiment/list
- GET  /experiment/{run_id}/activations?from_step=&to_step=
- POST /experiment/start-demo  (start mock run)
- POST /experiment/{run_id}/stop
- WS   /ws/experiment/{run_id}/stream
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
from typing import List
from . import storage, experiments
from .models import ActivationSnapshot
import asyncio
import json

router = APIRouter()


@router.get("/experiment/list")
async def list_experiments():
    """
    Return list of registered experiments metadata.
    """
    exps = storage.list_experiments()
    # translate SQLModel Experiment -> simple dict
    out = [{"id": e.run_id, "name": e.name, "created_at": e.created_at, "model_type": e.model_type} for e in exps]
    return JSONResponse(content=out)


@router.get("/experiment/{run_id}/activations")
async def get_activations(run_id: str, from_step: int = 0, to_step: int = 0):
    """
    Return array of ActivationSnapshot for given run and step range.
    """
    snaps = experiments.get_snapshots(run_id, from_step=from_step, to_step=to_step)
    # Pydantic models are JSON serializable via .model_dump
    return [s.model_dump() for s in snaps]


@router.post("/experiment/start-demo")
async def start_demo(background: BackgroundTasks):
    """
    Start a demo/mock experiment. Returns run_id.
    """
    run_id = experiments.start_mock_run(name="demo-run", model_type="memnet")
    return {"run_id": run_id}


@router.post("/experiment/{run_id}/stop")
async def stop_run(run_id: str):
    """
    Stop a mock run if one exists.
    """
    experiments.stop_mock_run(run_id)
    return {"stopped": True}


# WebSocket manager helpers: we need per-run set of connected websockets
_ws_connections: dict[str, List[WebSocket]] = {}


async def _send_json_safe(ws: WebSocket, payload):
    """
    Send JSON to websocket and handle disconnects.
    """
    try:
        await ws.send_json(payload)
    except Exception:
        # ignore send errors — caller will cleanup on disconnect
        pass


@router.websocket("/ws/experiment/{run_id}/stream")
async def ws_stream(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint. On connect:
      - Accept connection
      - Register a callback that will schedule sending of snapshots to this websocket
      - Keep websocket alive until client disconnects
    """

    await websocket.accept()
    _ws_connections.setdefault(run_id, []).append(websocket)

    # define a callback used by experiments.subscribe that schedules sending to ws
    loop = asyncio.get_running_loop()

    def cb(snapshot: ActivationSnapshot):
        """
        Schedule snapshot to be sent to websocket in the event loop.
        """
        try:
            payload = snapshot.model_dump()
            # schedule sending without awaiting here (send in background)
            asyncio.run_coroutine_threadsafe(_send_json_safe(websocket, payload), loop)
        except Exception as e:
            print(f"Callback send error: {e}")

    # subscribe
    experiments.subscribe(run_id, cb)

    try:
        # keep websocket open; respond to ping/pong from client
        while True:
            # simple receive to detect client disconnect; timeout not necessary
            try:
                data = await websocket.receive_text()
                # echo or ignore; clients can send pings
            except WebSocketDisconnect:
                break
            except Exception:
                # other exceptions ignored
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        # cleanup subscription and connection
        experiments.unsubscribe(run_id, cb)
        if run_id in _ws_connections and websocket in _ws_connections[run_id]:
            _ws_connections[run_id].remove(websocket)
