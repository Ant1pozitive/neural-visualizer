import React from "react";
import { useExperimentStore } from "../store/experimentStore";

const TimeSlider: React.FC = () => {
  const { state, currentStep, setStep } = useExperimentStore();

  if (!state) return null;

  return (
    <input
      type="range"
      min={0}
      max={state.steps - 1}
      value={currentStep}
      onChange={(e) => setStep(Number(e.target.value))}
      className="time-slider"
    />
  );
};

export default TimeSlider;
