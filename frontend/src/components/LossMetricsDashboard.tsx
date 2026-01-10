// src/components/LossMetricsDashboard.tsx
import React from 'react';
import PlotlyWrapper from '../utils/plotlyWrapper';

/**
 * LossMetricsDashboard renders loss + additional metric series.
 * Consumer passes an array of { step, loss, [metrics] } from backend snapshots.
 */

type SeriesPoint = { step: number; loss?: number; metrics?: Record<string, number> };

type Props = {
  series: SeriesPoint[];
};

export default function LossMetricsDashboard({ series }: Props) {
  const steps = series.map((s) => s.step);
  const losses = series.map((s) => s.loss ?? NaN);

  // collect metric names
  const metricNames = React.useMemo(() => {
    const set = new Set<string>();
    for (const s of series) {
      if (s.metrics) for (const k of Object.keys(s.metrics)) set.add(k);
    }
    return Array.from(set);
  }, [series]);

  const data: any[] = [
    { x: steps, y: losses, type: 'scatter', mode: 'lines', name: 'Loss' }
  ];

  for (const name of metricNames) {
    data.push({
      x: steps,
      y: series.map((s) => (s.metrics && name in s.metrics ? s.metrics[name] : NaN)),
      type: 'scatter',
      mode: 'lines',
      name
    });
  }

  return (
    <div className="panel">
      <h3 className="text-sm font-medium mb-2">Loss & Metrics</h3>
      <div style={{ height: 240 }}>
        <PlotlyWrapper data={data} layout={{ yaxis: { title: 'value' } }} />
      </div>
    </div>
  );
}
