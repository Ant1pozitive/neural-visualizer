// src/components/LayerSelector.tsx
import React from 'react';

type Props = {
  layers: number[];
  heads?: number[];
  onSelect: (layer: number, head?: number) => void;
};

/**
 * LayerSelector lets user pick a layer and optional head.
 * Minimal but usable UI to control which attention slice to display.
 */
export default function LayerSelector({ layers, heads = [], onSelect }: Props) {
  const [layer, setLayer] = React.useState<number>(layers[0] ?? 0);
  const [head, setHead] = React.useState<number | undefined>(heads[0]);

  React.useEffect(() => {
    onSelect(layer, head);
  }, [layer, head]);

  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs text-slate-600">Layer</label>
      <select value={layer} onChange={(e) => setLayer(Number(e.target.value))} className="w-full border rounded p-1">
        {layers.map((l) => (
          <option key={l} value={l}>
            Layer {l}
          </option>
        ))}
      </select>

      {heads.length > 0 && (
        <>
          <label className="text-xs text-slate-600">Head</label>
          <select
            value={head}
            onChange={(e) => setHead(e.target.value === '' ? undefined : Number(e.target.value))}
            className="w-full border rounded p-1"
          >
            <option value="">all</option>
            {heads.map((h) => (
              <option key={h} value={h}>
                Head {h}
              </option>
            ))}
          </select>
        </>
      )}
    </div>
  );
}
