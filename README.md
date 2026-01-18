# Interactive Neural Network Training Visualization

A research-grade, modular platform for instrumenting, tracing, storing and interactively visualizing internal states of deep learning models (attention, memory reads/writes, activations, gradients, reconstructions, metrics). The platform is model-agnostic (works with arbitrary PyTorch models) and includes:

* Frontend (Layer 1) — React + TypeScript UI to explore attention heatmaps, memory slots, read/write timelines, loss/metrics and reconstructions.
* API / Orchestration (Layer 2) — FastAPI service exposing REST endpoints and WebSocket streams to serve experiment metadata and live snapshots.
* Training & Tracing Engine (Layer 3) — Lightweight async Trainer that can instrument models, run training loops, and emit time-indexed snapshots.
* Instrumentation (Layer 4) — Non-intrusive `Instrumentor` with adapters (HuggingFace + generic) to attach hooks and convert outputs to consistent snapshots.
* Data & Storage (Layer 5) — SQLModel metadata + blob storage (local compressed NPZ, optional S3) for efficient persistent storage of heavy tensors and metadata.

This repository is intended for researchers and engineers who want to debug or explain complex models (especially memory-augmented or attention-heavy ones) during training and inference.

---

## Table of contents

* [Architecture](#architecture)
* [Prerequisites](#prerequisites)
* [Quickstart (local dev)](#quickstart-local-dev)
* [Project layout](#project-layout)
* [API reference (essential endpoints)](#api-reference-essential-endpoints)
* [WebSocket / Live streaming](#websocket--live-streaming)
* [Instrumenting your model (generic)](#instrumenting-your-model-generic)
* [Training integration (Trainer)](#training-integration-trainer)
* [Storage design & data model](#storage-design--data-model)
* [Production recommendations](#production-recommendations)
* [Troubleshooting](#troubleshooting)
* [Next steps / Roadmap](#next-steps--roadmap)
* [Contributing & License](#contributing--license)

---

## Architecture

High-level flow:

```
[Instrumented Model / Trainer]  --> (register_snapshot) --> [Storage: metadata + blobs]
                        ↘                                     ↘
                         → (WS publish) → [FastAPI WebSocket] → [Frontend (React)] 
                         → (REST)        → [FastAPI HTTP API]  → [Frontend requests]
```

Key design choices:

* **Model-agnostic instrumentation** — adapters + forward/backward hooks; no mandatory changes to model code.
* **Non-blocking I/O** — heavy I/O (writing blobs) is offloaded to threads/async tasks.
* **Lazy loading** — frontend first receives metadata; heavy blobs are fetched on demand.
* **Blob backend abstraction** — local NPZ by default; S3 optional for production.

---

## Prerequisites

* Python 3.10+ (recommended)
* Node.js 18+ / npm or pnpm
* For CPU-only PyTorch: install `torch` per official instructions (or GPU variant if available)
* Optional: `boto3` if you want S3 blob storage

Recommended Python packages are listed in `backend/requirements.txt`.

---

## Quickstart (local dev)

### 1. Backend — API & demo data

```bash
# from repository root
cd backend

# create virtualenv
python -m venv .venv
source .venv/bin/activate

# install dependencies (torch may need special instructions)
pip install -r requirements.txt

# start API (FastAPI/uvicorn)
./start.sh
# or: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Notes**

* On startup the backend runs a small auto-demo mock trainer (`auto-demo`) that emits snapshots to help rapid frontend testing.
* If you want to run the instrumented `Trainer` with a synthetic example, run:

```bash
python -m app.train_worker
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

By default the frontend expects the backend at `http://localhost:8000`. Adjust `VITE_API_BASE` in `frontend/.env` if necessary.

---

## Project layout

```
frontend/                      # Layer 1 (Presentation)
  src/
    components/                # UI components: AttentionHeatmap, MemorySlotsView, ...
    api/client.ts              # typed client to backend
backend/
  app/
    main.py                    # FastAPI app
    api.py                     # API router
    models.py                  # Pydantic/SQLModel schemas
    storage_core.py            # Layer 5 core storage API (SQLModel + blob refs)
    storage_blobs.py           # blob backends (LocalNPZ, optional S3)
    storage_utils.py           # path helpers
    experiments.py             # in-memory registry + mock trainers
    trace_adapter.py           # tensor -> TimeIndexedTensor conversion
    training_engine.py         # Trainer + TraceCollector
    instrumentation/
      instrumentor.py          # Instrumentor (attach/detach hooks)
      adapters.py              # HuggingFace + Generic adapters
      utils.py
    train_worker.py            # example training launcher
  requirements.txt
  start.sh
README.md                      # (this file)
```

---

## API reference (essential endpoints)

> Base URL: `http://localhost:8000`

* `GET /experiment/list`
  Returns array of experiment metadata: `[{ id: run_id, name, created_at, model_type }]`

* `GET /experiment/{run_id}/activations?from_step=&to_step=`
  Returns array of activation snapshot metadata (optionally for a range). By default returns snapshots persisted for the run. Snapshots contain blob references for heavy arrays.

* `POST /experiment/start-demo`
  Start a new mock demo run (returns `{"run_id": "<uuid>"}`).

* `POST /experiment/{run_id}/stop`
  Stop a mock run (if running).

* `WS  /ws/experiment/{run_id}/stream`
  WebSocket endpoint that streams live `ActivationSnapshot` JSON messages as training emits them.

* (Recommended/optional) `GET /experiment/{run_id}/snapshot/{snapshot_id}/blob?field=attention`
  Endpoint to stream a single blob (attention/read_weights/etc). If not present yet, frontend can use `storage_core.load_snapshot` internal logic or a custom API.

**Notes on payloads**

* `ActivationSnapshot` is a JSON object with:

  * `step` (int), `epoch` (int optional), `loss` (float optional), `metrics` (dict optional)
  * `attention`, `read_weights`, `write_weights`, `memory_slots`, `reconstructions` — each can be a `TimeIndexedTensor` or a blob reference (relative path or S3 URI).
* `TimeIndexedTensor`:

  * `time` (int optional), `shape` (array of ints), `buffer` (flattened array of numbers).

---

## WebSocket / Live streaming

Frontend can open a WebSocket to `ws://localhost:8000/ws/experiment/{run_id}/stream`. Each message is a JSON-serialized `ActivationSnapshot`. The WS is used for live trace visualization; persisted snapshots are written to storage by the same path.

Example (browser console):

```js
const ws = new WebSocket("ws://localhost:8000/ws/experiment/<run_id>/stream");
ws.onmessage = (ev) => {
  const snapshot = JSON.parse(ev.data);
  console.log("live snapshot", snapshot.step);
};
```

---

## Instrumenting your model (generic)

This system is model-agnostic. You have two options:

### A. Non-intrusive (recommended) — use `Instrumentor` (no source changes)

```py
from app.instrumentation import Instrumentor
from my_project.model import MyModel  # any nn.Module
model = MyModel(...)

run_id = "my-run-001"
instr = Instrumentor(model, run_id, signals=["attention","activations","gradients","memory"])
instr.attach()

# run training loop (Trainer or custom)
# instrumentor will schedule snapshots automatically
# when done:
instr.detach()
```

`Instrumentor` attaches forward/backward hooks and uses adapters to convert model outputs to `TimeIndexedTensor`s. It schedules background writes to storage and notifies subscribers via `experiments.register_snapshot`.

### B. Explicit instrumentation (fine-grained)

* If your model produces internal per-step read/write histories (e.g., MemNet), call `trace_adapter.to_timeindexed(tensor, time=step)` at the internal points in your model/training loop and then call `experiments.register_snapshot(run_id, snapshot)` to register snapshots with full control.

**Example snippet (MemNet-style):**

```py
from app.trace_adapter import to_timeindexed
from app.models import ActivationSnapshot
# inside training loop after model forward:
logits, recon, read_hist, write_hist = model(inputs, return_attn=True)
# choose per-inner-step or last-step:
last_read = read_hist[:, -1, ...]
read_ti = to_timeindexed(last_read, time=global_step)
snapshot = ActivationSnapshot(step=global_step, read_weights=read_ti, reconstructions=to_timeindexed(recon, time=global_step), loss=loss_val)
experiments.register_snapshot(run_id, snapshot)
```

---

## Training integration (Trainer)

Use `app.training_engine.Trainer` for a friendly async producer:

```py
from app.training_engine import Trainer
trainer = Trainer(model, optimizer, loss_fn, dataloader, run_id, device="cpu", num_epochs=5)
trainer.start_in_background()
# trainer registers snapshots via experiments.register_snapshot
```

The `Trainer` is generic and attempts `model(inputs, return_attn=True)` if model supports it. It also probes `model.memory` attributes automatically to capture memory slot tensors.

---

## Storage design & data model

* **Metadata DB** — `sqlmodel` (SQLite by default) stores `ExperimentMeta` and `SnapshotMeta` rows. `SnapshotMeta` includes `blob_paths` mapping field → blob reference and a `meta_json` for `loss`/`metrics`.
* **Blob storage** — heavy arrays are saved as compressed `.npz` files in `runs/{run_id}/blobs/` (or optionally uploaded to S3). Each blob has a small `.meta.json` sidecar containing `shape` and extra metadata.
* **Lazy-load** — frontend first receives metadata; heavy blobs are requested on demand via specialized API calls.

**Primary storage functions**

* `storage_core.persist_activation_snapshot(run_id, ActivationSnapshot)` — saves blobs and creates `SnapshotMeta`.
* `storage_core.list_snapshots(run_id, ...)` — query snapshot metadata.
* `storage_core.load_snapshot(...)` — load snapshot and optionally blobs.

**Retention & cleanup**

* `storage_core.cleanup_old_runs(retention_days=N)` — deletes old run artifacts.

---

## Production recommendations

1. **Blob store** — replace local NPZ with S3 (S3Backend included). Configure `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and bucket name.
2. **Database** — replace SQLite with PostgreSQL (connection string in `DB_URL`).
3. **Message queue** — use Kafka/RabbitMQ to decouple trainer → storage → websocket broadcasting for scalability.
4. **Authentication** — add JWT/OAuth with scope-based access to experiment data.
5. **Compression & sampling** — for huge models, store only summaries or sampled heads/slots. Consider saving only norms or PCA-reduced embeddings.
6. **Monitoring & retention** — integrate Prometheus/Grafana and define retention/cleanup policies for disk.
7. **Containerization** — provide Dockerfiles and `docker-compose` for reproducible deployment.

---

## Troubleshooting

* **Frontend cannot connect to backend**

  * Verify `VITE_API_BASE` in `frontend/.env`. Ensure CORS allowed origins in `app.main`.
* **WebSocket fails**

  * Use `ws://` for HTTP local dev; check port and firewall. Look for server logs for WS errors.
* **Large memory / disk usage**

  * Use NPZ compression; enable retention and cleanup; offload blobs to S3.
* **PyTorch installation issues**

  * Follow official instructions for platform & CUDA version: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## Next steps / Roadmap

* Add API endpoints for blob streaming and partial tensor requests (e.g., fetch a slice of attention).
* Add tests and CI for core pipelines (instrumentation, trainer, storage).
* Add `docker-compose.yml` to orchestrate frontend + backend + optionally MinIO (S3 emulator) for local testing.
* Add RBAC & authentication.
* Add per-inner-step emission mode for memory-augmented models like MemNet (option in Trainer).

---

## Contributing & License

* Fork the repo, create feature branches, run tests, and open PRs.
* Code style: Python — `black` + `isort`; TypeScript — `prettier`.
* Add unit tests for instrumentor and storage backends.

**MIT License**
