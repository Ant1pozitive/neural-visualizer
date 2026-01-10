// src/components/AttentionHeatmap.tsx
import React from 'react';
import PlotlyWrapper from '../utils/plotlyWrapper';
import type { TimeIndexedTensor } from '../types';

type Props = {
  title?: string;
  attention?: TimeIndexedTensor | null;
  step?: number;
};

/**
 * AttentionHeatmap renders a 2D attention matrix for a specific step.
 * The backend exposes attention as a TimeIndexedTensor with shape like [heads, seq_q, seq_k]
 * or [seq, seq] depending on aggregation.
 */
export default function AttentionHeatmap({ title = 'Attention', attention, step }: Props) {
  const matrix: number[][] = React.useMemo(() => {
    if (!attention || !attention.buffer || attention.buffer.length === 0) return [[]];
    const shape = attention.shape;
    // if 2D matrix (seq, seq)
    if (shape.length === 2) {
      const [r, c] = shape;
      const buf = attention.buffer;
      const out: number[][] = [];
      for (let i = 0; i < r; i++) {
        out.push(buf.slice(i * c, i * c + c));
      }
      return out;
    }
    // if 3D [heads, q, k] -> merge heads by averaging
    if (shape.length === 3) {
      const [h, q, k] = shape;
      const buf = attention.buffer;
      const agg = Array.from({ length: q }, () => Array(k).fill(0));
      for (let hh = 0; hh < h; hh++) {
        const offset = hh * q * k;
        for (let i = 0; i < q; i++) {
          for (let j = 0; j < k; j++) {
            agg[i][j] += buf[offset + i * k + j];
          }
        }
      }
      for (let i = 0; i < q; i++) {
        for (let j = 0; j < k; j++) {
          agg[i][j] /= h;
        }
      }
      return agg;
    }
    return [[]];
  }, [attention]);

  const data = [
    {
      z: matrix,
      type: 'heatmap' as const,
      hoverongaps: false,
      zsmooth: 'best' as const,
      colorbar: { title: 'attention' }
    }
  ];

  const layout = {
    title: `${title}${typeof step === 'number' ? ` — step ${step}` : ''}`,
    xaxis: { title: 'Key position' },
    yaxis: { title: 'Query position' }
  };

  return (
    <div className="panel">
      <PlotlyWrapper data={data} layout={layout} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
