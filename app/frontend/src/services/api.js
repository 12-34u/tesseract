/**
 * API client for the Tesseract backend. Errors are thrown, not swallowed,
 * so the dashboard can show that data failed to load.
 */

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchJson(endpoint) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);
  if (!response.ok) {
    throw new Error(`${endpoint}: HTTP ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  getSummary: () => fetchJson('/api/summary'),
  getCrucible: () => fetchJson('/api/crucible'),
  getKScaling: () => fetchJson('/api/k-scaling'),
  getCellularAutomaton: () => fetchJson('/api/cellular-automaton'),
  getRawResults: () => fetchJson('/api/raw-results'),
  // Phase 2. Independent of the Phase 1 endpoints: the status payload reports
  // pending states for artifacts that do not exist yet.
  getPhase2Status: () => fetchJson('/api/phase2/status'),
  getPhase2Gates: () => fetchJson('/api/phase2/gates'),
  getPhase2P1b: () => fetchJson('/api/phase2/p1b'),
  getPhase2Results: () => fetchJson('/api/phase2/results'),
};
