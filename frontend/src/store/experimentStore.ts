import { create } from "zustand";
import { ExperimentState } from "../types/experiment";

interface ExperimentStore {
  state: ExperimentState | null;
  currentStep: number;
  setState: (state: ExperimentState) => void;
  setStep: (step: number) => void;
}

export const useExperimentStore = create<ExperimentStore>((set) => ({
  state: null,
  currentStep: 0,
  setState: (state) => set({ state }),
  setStep: (step) => set({ currentStep: step })
}));
