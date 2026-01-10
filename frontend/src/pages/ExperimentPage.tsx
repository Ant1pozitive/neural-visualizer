import React from "react";
import TimeSlider from "../components/TimeSlider";
import MetricChart from "../components/MetricChart";
import AttentionHeatmap from "../components/AttentionHeatmap";
import MemorySlotsView from "../components/MemorySlotsView";
import ReadWriteTimeline from "../components/ReadWriteTimeline";

const ExperimentPage: React.FC = () => {
  return (
    <div className="experiment-page">
      <TimeSlider />
      <MetricChart />
      <AttentionHeatmap />
      <MemorySlotsView />
      <ReadWriteTimeline />
    </div>
  );
};

export default ExperimentPage;
