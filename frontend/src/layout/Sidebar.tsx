import React from "react";

const Sidebar: React.FC = () => {
  return (
    <aside className="sidebar">
      <h2>Neural Visualizer</h2>
      <nav>
        <ul>
          <li>Experiment</li>
          <li>Metrics</li>
          <li>Memory</li>
          <li>Attention</li>
        </ul>
      </nav>
    </aside>
  );
};

export default Sidebar;
