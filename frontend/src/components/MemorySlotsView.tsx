// src/components/MemorySlotsView.tsx
import React from 'react';
import type { TimeIndexedTensor } from '../types';

/**
 * MemorySlotsView provides a compact visualization of memory slot states.
 * For simplicity we visualize L2 norms of slot vectors as a bar chart.
 */

type Props = {
  memory?: TimeIndexedTensor | null;
  title?: string;
  step?: number;
};

export default function MemorySlotsView({ memory, title = 'Memory Slots', step }: Props) {
  const norms = React.useMemo(() => {
    if (!memory || !memory.buffer || memory.buffer.length === 0) return [];
    const shape = memory.shape; // [slots, dim] or [batch, slots, dim]
    let slots = 0;
    let dim = 0;
    let buf = memory.buffer;
    if (shape.length === 2) {
      [slots, dim] = shape;
    } else if (shape.length === 3) {
      // reduce batch by picking first element
      const [batch, s, d] = shape;
      slots = s;
      dim = d;
      buf = buf.slice(0, s * d);
    } else {
      return [];
    }
    const out: number[] = [];
    for (let i = 0; i < slots; i++) {
      let sum = 0;
      for (let j = 0; j < dim; j++) {
        const v = buf[i * dim + j];
        sum += v * v;
      }
      out.push(Math.sqrt(sum));
    }
    return out;
  }, [memory]);

  return (
    <div className="panel h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium">{title}{typeof step === 'number' ? ` — step ${step}` : ''}</h3>
      </div>
      <div className="flex-1 overflow-auto">
        <ul className="divide-y">
          {norms.map((n, idx) => (
            <li key={idx} className="py-2 flex items-center gap-3">
              <div className="w-12 text-xs text-slate-500">slot {idx}</div>
              <div className="flex-1 bg-slate-100 rounded-full h-3 overflow-hidden">
                <div style={{ width: `${Math.min(100, n * 10)}%` }} className="h-full bg-sky-500" />
              </div>
              <div className="w-12 text-right text-xs">{n.toFixed(3)}</div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
