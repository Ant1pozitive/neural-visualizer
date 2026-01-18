# Instrumentation package

## Goals
Provide non-intrusive hooks to capture activations, attentions, memory and gradients
from arbitrary PyTorch models and emit ActivationSnapshots to the experiments subsystem.

## Quick example

```py
from app.instrumentation import Instrumentor
from my_project.model import MyModel

model = MyModel(...)
run_id = "your-run-id"
instr = Instrumentor(model, run_id, signals=["attention","activations","gradients","memory"])
instr.attach()

# run training loop (or use Trainer from training_engine)
# instrumentor will schedule snapshots asynchronously
# when done:
instr.detach()
```
