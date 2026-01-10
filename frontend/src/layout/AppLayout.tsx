import React from "react";
import Sidebar from "./Sidebar";

const AppLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="app-root">
      <Sidebar />
      <main className="app-main">{children}</main>
    </div>
  );
};

export default AppLayout;
