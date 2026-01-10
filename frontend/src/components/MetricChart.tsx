import React, { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import { useExperimentStore } from "../store/experimentStore";

const MetricChart: React.FC = () => {
  const ref = useRef<HTMLDivElement>(null);
  const state = useExperimentStore((s) => s.state);

  useEffect(() => {
    if (!ref.current || !state) return;

    Plotly.newPlot(ref.current, [
      {
        y: state.loss,
        type: "scatter",
        mode: "lines",
        name: "Loss"
      }
    ]);
  }, [state]);

  return <div ref={ref} />;
};

export default MetricChart;
