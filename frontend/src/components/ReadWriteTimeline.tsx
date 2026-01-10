import React from "react";
import { useExperimentStore } from "../store/experimentStore";

const ReadWriteTimeline: React.FC = () => {
  const { state } = useExperimentStore();
  if (!state) return null;

  return (
    <div className="rw-timeline">
      <h4>Read / Write Activity</h4>
      <p>Read weights shape: {state.readWeights.length}</p>
      <p>Write weights shape: {state.writeWeights.length}</p>
    </div>
  );
};

export default ReadWriteTimeline;
