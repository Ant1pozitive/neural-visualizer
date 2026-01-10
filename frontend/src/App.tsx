// src/App.tsx
import React from 'react';
import Layout from './components/Layout';
import AttentionHeatmap from './components/AttentionHeatmap';
import MemorySlotsView from './components/MemorySlotsView';
import ReadWriteTimeline from './components/ReadWriteTimeline';
import LossMetricsDashboard from './components/LossMetricsDashboard';
import ReconstructionInspector from './components/ReconstructionInspector';
import LayerSelector from './components/LayerSelector';
import TimeSlider from './components/TimeSlider';
import { fetchExperiments, fetchActivations, createExperimentSocket } from './api/client';
import type { ExperimentMeta, ActivationSnapshot } from './types';

/**
 * Root App component orchestrates fetching experiments, selecting
 * runs and stepping through activations. It wires components together.
 *
 * This App is intentionally minimal on state management; for larger projects
 * move state into a separate store (e.g. Zustand or Redux).
 */

export default function App() {
  const [experiments, setExperiments] = React.useState<ExperimentMeta[]>([]);
  const [selected, setSelected] = React.useState<string | null>(null);
  const [snapshots, setSnapshots] = React.useState<ActivationSnapshot[]>([]);
  const [step, setStep] = React.useState<number>(0);
  const [liveSocket, setLiveSocket] = React.useState<WebSocket | null>(null);

  // fetch experiments on mount
  React.useEffect(() => {
    (async () => {
      try {
        const list = await fetchExperiments();
        setExperiments(list);
        if (list.length > 0) {
          setSelected(list[0].id);
        }
      } catch (err) {
        console.error('Failed to fetch experiments', err);
      }
    })();
  }, []);

  // when selected run changes, fetch activations (initial window)
  React.useEffect(() => {
    if (!selected) return;
    (async () => {
      try {
        const acts = await fetchActivations(selected, 0, 100);
        setSnapshots(acts);
        setStep(acts.length > 0 ? acts[0].step : 0);
      } catch (err) {
        console.error('Failed to fetch activations', err);
      }
    })();

    // open websocket for live updates, if available
    try {
      const ws = createExperimentSocket(selected);
      ws.onopen = () => console.info('WS open');
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as ActivationSnapshot;
          // append live snapshot
          setSnapshots((prev) => [...prev, data]);
        } catch (e) {
          console.error('WS parse error', e);
        }
      };
      ws.onerror = (e) => console.error('WS error', e);
      ws.onclose = () => console.info('WS closed');

      setLiveSocket(ws);

      return () => {
        ws.close();
        setLiveSocket(null);
      };
    } catch (e) {
      console.warn('WebSocket not available', e);
    }
  }, [selected]);

  const current = React.useMemo(() => {
    // find snapshot for current step or last snapshot
    if (snapshots.length === 0) return null;
    const s = snapshots.find((x) => x.step === step);
    return s ?? snapshots[snapshots.length - 1];
  }, [snapshots, step]);

  // simulated layer/head info (in real app extract from model config)
  const layers = React.useMemo(() => Array.from({ length: 6 }, (_, i) => i), []);
  const heads = React.useMemo(() => Array.from({ length: 8 }, (_, i) => i), []);

  // handler for layer selection (not wired to backend in MVP)
  const handleLayerSelect = (layer: number, head?: number) => {
    console.info('Selected layer', layer, 'head', head);
    // in full implementation pass selection to backend query params
  };

  return (
    <Layout title="Interactive NN Training Visualization">
      {/* left column: attention + memory */}
      <AttentionHeatmap attention={current?.attention ?? null} step={current?.step} />
      <MemorySlotsView memory={current?.memory_slots ?? null} step={current?.step} />
      {/* right column: timelines + recon */}
      <ReadWriteTimeline readWeights={current?.read_weights ? [current.read_weights] : []} writeWeights={current?.write_weights ? [current.write_weights] : []} />
      <div className="space-y-4">
        <LossMetricsDashboard series={snapshots.map((s) => ({ step: s.step, loss: s.loss, metrics: s.metrics }))} />
        <ReconstructionInspector recon={current?.reconstructions ?? null} step={current?.step} />
        <div className="flex gap-3 items-center">
          <div className="flex-1">
            <TimeSlider min={snapshots[0]?.step ?? 0} max={snapshots[snapshots.length - 1]?.step ?? 0} step={step} onChange={(v) => setStep(v)} />
          </div>
          <div className="w-80">
            <LayerSelector layers={layers} heads={heads} onSelect={handleLayerSelect} />
          </div>
        </div>
      </div>
    </Layout>
  );
}
