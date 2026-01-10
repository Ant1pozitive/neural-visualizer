export interface ExperimentState {
  steps: number;
  loss: number[];
  attention: number[][][];
  readWeights: number[][];
  writeWeights: number[][];
  memorySlots: number[][][];
}
