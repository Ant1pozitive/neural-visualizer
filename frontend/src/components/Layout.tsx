// src/components/Layout.tsx
import React from 'react';
import { motion } from 'framer-motion';

type Props = {
  title?: string;
  children: React.ReactNode;
};

/**
 * Global layout component used by the application.
 * Provides left control panel and main content area.
 */
export default function Layout({ title = 'NN Visualization', children }: Props) {
  return (
    <div className="h-screen flex flex-col">
      <header className="h-16 flex items-center px-6 shadow-sm bg-white">
        <h1 className="text-lg font-semibold">{title}</h1>
      </header>

      <main className="flex-1 flex overflow-hidden p-6 gap-6">
        <aside className="w-80 flex-shrink-0 space-y-4">
          <div className="panel">
            <h2 className="text-sm font-medium mb-2">Experiment Controls</h2>
            <div className="space-y-3">
              <div className="text-xs text-slate-500">Select experiment, model and time window.</div>
            </div>
          </div>
          <div className="panel">
            <h3 className="text-sm font-medium mb-2">Layers & Heads</h3>
            <div>{/* Placeholder: LayerSelector will be injected by App */}</div>
          </div>
        </aside>

        <section className="flex-1 grid grid-cols-2 grid-rows-2 gap-6">
          {children}
        </section>
      </main>

      <footer className="h-12 flex items-center justify-center text-xs text-slate-500">
        Built for research-grade model visualization.
      </footer>
    </div>
  );
}
