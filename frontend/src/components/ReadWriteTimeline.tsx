// src/components/ReadWriteTimeline.tsx
import React from 'react';
import PlotlyWrapper from '../utils/plotlyWrapper';
import type { TimeIndexedTensor } from '../types';

type Props = {
  readWeights?: TimeIndexedTensor[] | null;
  writeWeights?: TimeIndexedTensor[] | null;
  title?: string;
};

/**
 * ReadWriteTimeline visualizes aggregated read and write strengths over time.
 * It accepts arrays of TimeIndexedTensor snapshots (one per step).
 * For efficiency, we aggregate by taking max across query/key dimension and averaging across batch/heads.
 */

function aggregateStrengths(tensors?: TimeIndexedTensor[] | null) {
  if (!tensors || tensors.length === 0) return [];
  const out: number[] = [];
  for (const t of tensors) {
    // compute a single scalar strength per snapshot: mean of abs(buffer)
    const buf = t.buffer;
    if (!buf || buf.length === 0) {
      out.push(0);
      continue;
    }
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += Math.abs(buf[i]);
    out.push(sum / buf.length);
  }
  return out;
}

export default function ReadWriteTimeline({ readWeights, writeWeights, title = 'Read/Write Activity' }: Props) {
  const readSeries = React.useMemo(() => aggregateStrengths(readWeights), [readWeights]);
  const writeSeries = React.useMemo(() => aggregateStrengths(writeWeights), [writeWeights]);

  const x = Array.from({ length: Math.max(readSeries.length, writeSeries.length) }, (_, i) => i);

  const data = [
    { x, y: readSeries, type: 'scatter', mode: 'lines+markers', name: 'Read strength' },
    { x, y: writeSeries, type: 'scatter', mode: 'lines+markers', name: 'Write strength' }
  ];

  return (
    <div className="panel">
      <h3 className="text-sm font-medium mb-2">{title}</h3>
      <div style={{ height: 220 }}>
        <PlotlyWrapper data={data} layout={{ yaxis: { title: 'avg magnitude' } }} />
      </div>
    </div>
  );
}
