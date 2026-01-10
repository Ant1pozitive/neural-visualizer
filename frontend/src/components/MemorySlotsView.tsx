import React from "react";
import { useExperimentStore } from "../store/experimentStore";

const MemorySlotsView: React.FC = () => {
  const { state, currentStep } = useExperimentStore();
  if (!state) return null;

  return (
    <div className="memory-view">
      {state.memorySlots[currentStep].map((slot, i) => (
        <div key={i} className="memory-slot">
          {slot.map((v, j) => (
            <span key={j}>{v.toFixed(2)} </span>
          ))}
        </div>
      ))}
    </div>
  );
};

export default MemorySlotsView;
