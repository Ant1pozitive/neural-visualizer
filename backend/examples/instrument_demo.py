# backend/examples/instrument_demo.py
import asyncio
import torch
from torch import nn
from app.instrumentation import Instrumentor
from app.storage import register_experiment, init_db

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(32, 10)

    def forward(self, x):
        a = self.l1(x)
        b = self.relu(a)
        out = self.l2(b)
        # return tuple to let adapter see tensors
        return out, b

async def main():
    init_db()
    run = register_experiment(run_id="instr-demo-1", name="instr-demo", model_type="small")
    run_id = run.run_id
    m = SmallModel()
    instr = Instrumentor(m, run_id, signals=["activations","gradients"])
    instr.attach()

    # trivial training loop
    opt = torch.optim.Adam(m.parameters())
    loss_fn = nn.CrossEntropyLoss()
    for step in range(1, 20):
        inp = torch.randn(8, 16)
        target = torch.randint(0, 10, (8,))
        out = m(inp)
        logits = out[0] if isinstance(out, (list, tuple)) else out
        loss = loss_fn(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        await asyncio.sleep(0.01)  # yield to event loop for background tasks

    instr.detach()

if __name__ == "__main__":
    asyncio.run(main())
