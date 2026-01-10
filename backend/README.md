# Backend — Visualization API & Orchestration

## Run locally

1. Create venv and install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start server:
   ```bash
   ./start.sh
   ```

3. Backend will run on http://localhost:8000
* GET /experiment/list
* GET /experiment/{id}/activations?from_step=&to_step=
* WS /ws/experiment/{id}/stream
