// src/components/TimeSlider.tsx
import React from 'react';

/**
 * TimeSlider is a reusable slider component to navigate through steps/epochs.
 * It accepts min/max and current step and notifies parent on change.
 */
type Props = {
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
};

export default function TimeSlider({ min, max, step, onChange }: Props) {
  return (
    <div className="panel">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium">Time Slider</h3>
        <div className="text-xs text-slate-500">step {step}</div>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        value={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
