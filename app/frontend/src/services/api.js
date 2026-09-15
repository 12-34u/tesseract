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
};
