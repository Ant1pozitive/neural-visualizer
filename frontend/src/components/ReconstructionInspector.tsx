// src/components/ReconstructionInspector.tsx
import React from 'react';
import PlotlyWrapper from '../utils/plotlyWrapper';
import type { TimeIndexedTensor } from '../types';

type Props = {
  recon?: TimeIndexedTensor | null;
  title?: string;
  step?: number;
};

/**
 * ReconstructionInspector visualizes reconstructed embeddings.
 * For simplicity we project the embedding vector to its first N dimensions
 * and show them as a line (per token if tokenized). Backend can supply reconstructions per token.
 */

export default function ReconstructionInspector({ recon, title = 'Reconstruction', step }: Props) {
  if (!recon || recon.buffer.length === 0) {
    return (
      <div className="panel">
        <h3 className="text-sm font-medium">Reconstruction</h3>
        <div className="text-xs text-slate-500">No reconstruction available for this step.</div>
      </div>
    );
  }

  // assume shape [seq, dim] or [batch, seq, dim]; take first token's embedding if batch present
  let buf = recon.buffer;
  let dim = 0;
  let seq = 1;
  if (recon.shape.length === 2) {
    [seq, dim] = recon.shape;
  } else if (recon.shape.length === 3) {
    // [batch, seq, dim] -> take first batch
    const [batch, s, d] = recon.shape;
    seq = s;
    dim = d;
    buf = buf.slice(0, s * d);
  } else {
    return null;
  }

  // visualize first token embedding (or average over tokens)
  const token0 = buf.slice(0, dim);
  const x = Array.from({ length: token0.length }, (_, i) => i);
  const data = [{ x, y: token0, type: 'scatter', mode: 'lines+markers', name: 'recon dim' }];

  return (
    <div className="panel">
      <h3 className="text-sm font-medium mb-2">{title}{typeof step === 'number' ? ` — step ${step}` : ''}</h3>
      <div style={{ height: 220 }}>
        <PlotlyWrapper data={data} layout={{ xaxis: { title: 'embedding dim' }, yaxis: { title: 'value' } }} />
      </div>
    </div>
  );
}
