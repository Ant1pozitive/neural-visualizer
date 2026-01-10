// src/api/client.ts
import axios from 'axios';
import type { ExperimentMeta, ActivationSnapshot } from '../types';

/**
 * Simple typed API client for the Visualization API.
 * Adjust API_BASE if your backend runs on a different host or port.
 */
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000
});

/**
 * Fetch list of experiments (metadata).
 */
export async function fetchExperiments(): Promise<ExperimentMeta[]> {
  const res = await api.get('/experiment/list');
  return res.data as ExperimentMeta[];
}

/**
 * Fetch activations for a run / step range.
 * Backend should accept query params: ?from_step=&to_step=
 */
export async function fetchActivations(
  runId: string,
  fromStep = 0,
  toStep = 0
): Promise<ActivationSnapshot[]> {
  const res = await api.get(`/experiment/${encodeURIComponent(runId)}/activations`, {
    params: { from_step: fromStep, to_step: toStep }
  });
  return res.data as ActivationSnapshot[];
}

/**
 * Subscribe to WebSocket stream for live data.
 * The backend must expose ws:// or wss:// endpoint at /ws/experiment/:id/stream
 * This helper returns a WebSocket instance; caller should manage messages.
 */
export function createExperimentSocket(runId: string): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  // if backend on different host, change accordingly
  const host = new URL(API_BASE).host;
  const wsUrl = `${protocol}://${host}/ws/experiment/${encodeURIComponent(runId)}/stream`;
  return new WebSocket(wsUrl);
}
