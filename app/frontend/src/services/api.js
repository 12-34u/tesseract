/**
 * API Service for Tesseract Backend.
 * Connects to FastAPI backend at http://localhost:8000
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchJson(endpoint) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API Fetch Error [${endpoint}]:`, error);
    return { available: false, error: error.message };
  }
}

export const api = {
  getHealth: () => fetchJson('/api/health'),
  getSummary: () => fetchJson('/api/summary'),
  getCrucible: () => fetchJson('/api/crucible'),
  getKScaling: () => fetchJson('/api/k-scaling'),
  getCellularAutomaton: () => fetchJson('/api/cellular-automaton'),
  getExperiments: () => fetchJson('/api/experiments'),
  getRawResults: () => fetchJson('/api/raw-results'),
};
