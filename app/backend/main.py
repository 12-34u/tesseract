"""FastAPI Backend for Tesseract Evaluation Dashboard.

Exposes REST APIs reading experiment artifacts from runs/.
"""

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.results_loader import ResultsLoader

app = FastAPI(
    title="Tesseract Evaluation Dashboard API",
    description="Backend service serving Tesseract ML research prototype evaluation metrics.",
    version="1.0.0",
)

# Enable CORS for local Vite frontend (port 5173 / 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = ResultsLoader()


@app.get("/")
def root():
    return {"message": "Tesseract Evaluation Dashboard API", "status": "running"}


@app.get("/api/health")
def get_health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "loader": "ready",
    }


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
