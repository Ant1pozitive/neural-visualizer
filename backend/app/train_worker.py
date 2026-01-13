# app/train_worker.py
"""
Simple launcher script that creates a small synthetic dataset and runs Trainer.

This script is useful for local development or as an example how to start an instrumented
training job programmatically. It uses purely synthetic data and a small model if
a real model (e.g., MemNet) is not provided.

Usage:
    python -m app.train_worker

If you want to run an actual MemNet from your `memory-is-all-you-need` repo,
import it here and pass the model instance to Trainer instead of the synthetic model.
"""

import asyncio
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .training_engine import Trainer
from .experiments import start_mock_run  # convenience: registers experiment (but MockTrainer also starts)
from . import storage, experiments

# Small synthetic model that mimics a model returning (logits, recon, read_w_hist, write_w_hist)
class SyntheticMemLikeModel(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=64, seq_len=16, heads=4, slots=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(nn.Linear(embed_dim * seq_len, 256), nn.ReLU(), nn.Linear(256, 128))
        # emulate memory module: a tensor that changes
        self.register_buffer("memory_slots", torch.randn(slots, embed_dim))
        self.heads = heads
        self.slots = slots
        self.seq_len = seq_len

    def forward(self, x, return_attn: bool = False):
        """
        x: [B, seq_len] int tokens
        returns:
            logits [B, vocab_size] (toy)
            recon [B, seq_len, dim] (toy)
            read_w_hist [B, T, heads, slots] (here T==1 for simplicity)
            write_w_hist [B, T, heads, slots]
        """
        B = x.shape[0]
        emb = self.embed(x)  # [B, seq_len, dim]
        flat = emb.view(B, -1)
        logits = self.encoder(flat)  # [B, 128] as toy logits
        # produce recon as emb with noise
        recon = emb + 0.01 * torch.randn_like(emb)
        if return_attn:
            # produce pseudo read/write weights
            # here we create shape [B, 1, heads, slots] so that Trainer picks last timestep
            read_w = torch.abs(torch.randn(B, 1, self.heads, self.slots))
            write_w = torch.abs(torch.randn(B, 1, self.heads, self.slots))
            return logits, recon, read_w, write_w
        return logits

async def main():
    # ensure DB exists
    storage.init_db()

    # register experimental run (alternatively use start_mock_run to create mock trainer)
    run_id = str("train-demo-" + str(int(time.time())))

    # store metadata
    storage.register_experiment(run_id=run_id, name="training-demo", model_type="synthetic")

    # create synthetic dataset
    B = 32
    seq_len = 16
    vocab_size = 128
    total_samples = 1000

    inputs = torch.randint(0, vocab_size, (total_samples, seq_len), dtype=torch.long)
    # dummy targets for classification/regression (not used here)
    targets = torch.randint(0, 128, (total_samples,), dtype=torch.long)

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # model, optimizer, loss
    model = SyntheticMemLikeModel(vocab_size=vocab_size, embed_dim=64, seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dataloader=dataloader,
        run_id=run_id,
        device="cpu",
        num_epochs=2,
        emit_interval=1,
        max_steps=200
    )

    # start training in background and wait for completion
    task = trainer.start_in_background()
    try:
        await task
    except Exception as e:
        print("Training task failed:", e)

if __name__ == "__main__":
    import time
    asyncio.run(main())
