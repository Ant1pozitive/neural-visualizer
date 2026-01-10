import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useExperimentStore } from "../store/experimentStore";

const AttentionHeatmap: React.FC = () => {
  const ref = useRef<SVGSVGElement>(null);
  const { state, currentStep } = useExperimentStore();

  useEffect(() => {
    if (!ref.current || !state) return;

    const matrix = state.attention[currentStep];
    const size = 300;
    const cell = size / matrix.length;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const scale = d3.scaleSequential(d3.interpolateInferno).domain([0, 1]);

    matrix.forEach((row, i) => {
      row.forEach((v, j) => {
        svg
          .append("rect")
          .attr("x", j * cell)
          .attr("y", i * cell)
          .attr("width", cell)
          .attr("height", cell)
          .attr("fill", scale(v));
      });
    });
  }, [state, currentStep]);

  return <svg ref={ref} width={300} height={300} />;
};

export default AttentionHeatmap;
