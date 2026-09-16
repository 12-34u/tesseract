"""FastAPI backend for the Tesseract evaluation dashboard.

Read-only REST API over experiment artifacts in the runs directory.
Run from the repository root:

    PYTHONPATH=app/backend python -m uvicorn main:app --host 127.0.0.1 --port 8000

Environment:
    TESSERACT_RUNS_DIR      artifact directory (default: <repo>/runs)
    TESSERACT_CORS_ORIGINS  comma-separated allowed origins (default: Vite dev server)
"""

import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.phase2_loader import Phase2Loader
from services.results_loader import ResultsLoader

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

app = FastAPI(
    title="Tesseract Evaluation Dashboard API",
    description="Read-only view of Tesseract experiment artifacts.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("TESSERACT_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if o.strip()],
    allow_methods=["GET"],
    allow_headers=["*"],
)

loader = ResultsLoader()
# Phase 2 reads the same runs root. It is a separate loader so that a Phase 2
# artifact problem can never affect the Phase 1 endpoints.
phase2 = Phase2Loader(loader.runs_root)


@app.get("/api/health")
def get_health():
    return {"status": "ok", "timestamp": time.time(), "runs_root": str(loader.runs_root)}


@app.get("/api/summary")
def get_summary():
    return loader.get_summary()


@app.get("/api/crucible")
def get_crucible():
    return loader.get_crucible()


@app.get("/api/k-scaling")
def get_k_scaling():
    return loader.get_k_scaling()


@app.get("/api/cellular-automaton")
def get_cellular_automaton():
    return loader.get_cellular_automaton()


@app.get("/api/experiments")
def get_experiments():
    return loader.get_experiments()


@app.get("/api/raw-results")
def get_raw_results():
    return loader.get_raw_results()


# ============================================================================
# Phase 2 — pre-registration status and, once they exist, artifacts.
# These endpoints are independent of the Phase 1 endpoints above.
# ============================================================================


@app.get("/api/phase2/status")
def get_phase2_status():
    return phase2.get_status()


@app.get("/api/phase2/gates")
def get_phase2_gates():
    return phase2.get_gates()


@app.get("/api/phase2/p1b")
def get_phase2_p1b():
    return phase2.get_p1b()


@app.get("/api/phase2/results")
def get_phase2_results():
    return phase2.get_results()
