from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, invoices, analytics, ai_chat, reports, forecast
from app.database import engine
from app.models import models
from app.config import settings
import os

# Create all DB tables
models.Base.metadata.create_all(bind=engine)

# Create upload directory
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Finance Dashboard API",
    description="AI-powered finance automation — invoice processing, fraud detection, analytics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth")
app.include_router(invoices.router, prefix="/invoices")
app.include_router(analytics.router, prefix="/analytics")
app.include_router(ai_chat.router, prefix="/ai")
app.include_router(reports.router, prefix="/reports")
app.include_router(forecast.router, prefix="/analytics")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Finance Dashboard API is running"}
