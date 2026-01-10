// src/types.ts

/**
 * Shared TypeScript types for the frontend.
 * These types correspond to API payloads.
 */

export type ExperimentMeta = {
  id: string;
  name: string;
  created_at: string;
  model_type: string;
};

export type TimeIndexedTensor = {
  // Basic shape descriptor for a time-indexed tensor
  // buffer contains flattened numeric array (row-major)
  time: number;
  shape: number[]; // e.g. [batch, heads, seq, seq]
  buffer: number[]; // flattened values
};

export type ActivationSnapshot = {
  step: number;
  epoch?: number;
  attention?: TimeIndexedTensor; // attention weights for current step
  read_weights?: TimeIndexedTensor; // memnet read weights
  write_weights?: TimeIndexedTensor; // memnet write weights
  memory_slots?: TimeIndexedTensor; // memory slot vectors or norms
  loss?: number;
  metrics?: Record<string, number>;
  reconstructions?: TimeIndexedTensor;
};
