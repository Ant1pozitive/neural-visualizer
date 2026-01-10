import React from "react";
import AppLayout from "./layout/AppLayout";
import ExperimentPage from "./pages/ExperimentPage";

const App: React.FC = () => {
  return (
    <AppLayout>
      <ExperimentPage />
    </AppLayout>
  );
};

export default App;
