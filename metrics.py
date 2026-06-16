"""
LedgerAI — Prometheus Metrics Integration
==========================================
Add this to your FastAPI backend to expose metrics.

Install: pip install prometheus-fastapi-instrumentator

Add to main.py:
  from metrics import setup_metrics
  setup_metrics(app)
"""

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time

# ─── Custom Metrics ─────────────────────────────────────────

# Invoice metrics
invoices_processed = Counter(
    "ledgerai_invoices_processed_total",
    "Total number of invoices processed",
    ["status"]
)

anomalies_detected = Counter(
    "ledgerai_anomalies_detected_total",
    "Total number of anomalies detected by ML models"
)

# ML model metrics
ml_model_duration = Histogram(
    "ledgerai_ml_model_duration_seconds",
    "Time taken to run ML models",
    ["model_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

rag_query_duration = Histogram(
    "ledgerai_rag_query_duration_seconds",
    "Time taken for RAG pipeline queries",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

active_users = Gauge(
    "ledgerai_active_users",
    "Number of currently active users"
)

total_spend_gauge = Gauge(
    "ledgerai_total_spend_dollars",
    "Total spend amount in dollars"
)

# ─── Setup Function ─────────────────────────────────────────
def setup_metrics(app):
    """
    Call this in main.py to enable Prometheus metrics.
    
    Usage in main.py:
        from metrics import setup_metrics
        setup_metrics(app)
    """
    Instrumentator().instrument(app).expose(app)
    print("✅ Prometheus metrics enabled at /metrics")


# ─── Helper Functions ────────────────────────────────────────
def track_invoice_processed(status: str):
    """Call when an invoice is processed."""
    invoices_processed.labels(status=status).inc()


def track_anomaly_detected():
    """Call when an anomaly is detected."""
    anomalies_detected.inc()


def track_ml_model(model_name: str):
    """
    Context manager to track ML model execution time.
    
    Usage:
        with track_ml_model("random_forest"):
            run_random_forest()
    """
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            duration = time.time() - self.start
            ml_model_duration.labels(model_name=model_name).observe(duration)

    return Timer()


def track_rag_query():
    """Context manager to track RAG query time."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            duration = time.time() - self.start
            rag_query_duration.observe(duration)

    return Timer()


def update_total_spend(amount: float):
    """Update total spend gauge."""
    total_spend_gauge.set(amount)
