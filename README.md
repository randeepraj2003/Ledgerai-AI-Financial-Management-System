#  LedgerAI — AI-Powered Invoice & Financial Management System

> Full-stack AI platform that automates invoice processing, detects financial anomalies, forecasts spending, and enables intelligent document Q&A using RAG + LLM.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![React](https://img.shields.io/badge/React-18-cyan)
![ML Models](https://img.shields.io/badge/ML%20Models-6-orange)
![RAG](https://img.shields.io/badge/RAG-ChromaDB%20+%20Gemini-purple)

---

## Project Overview

LedgerAI is a production-ready financial intelligence platform built with modern AI/ML technologies. It combines traditional machine learning models with Generative AI to provide automated invoice processing, real-time anomaly detection, spending forecasts, and natural language Q&A over financial documents.

---

##  Features

### 🤖 AI & Machine Learning
- **ARIMA Time Series** — 3-month spending forecast
- **Random Forest Regressor** — Invoice amount prediction (MAE: $407)
- **Isolation Forest** — Unsupervised anomaly detection
- **DBSCAN Clustering** — Spending pattern discovery
- **K-Means Clustering** — Vendor risk scoring (HIGH/MEDIUM/LOW)
- **TF-IDF Cosine Similarity** — Duplicate invoice detection
- **NLP Categorizer** — Auto invoice categorization

### 🔍 RAG Pipeline
- PDF text extraction using PyMuPDF
- Vector embeddings using sentence-transformers
- ChromaDB vector database for similarity search
- Google Gemini LLM for intelligent Q&A
- Source citations with relevance scores

### 🧾 Invoice Management
- PDF drag & drop upload
- AI-powered auto extraction & categorization
- Approve / Reject workflow
- Fraud flag detection

### 🔐 Security
- JWT authentication
- Bcrypt password hashing
- CORS protection
- Environment variable management

### 📊 Monitoring & DevOps & Data Engineering
- Apache Airflow pipeline orchestration
- GitHub Actions CI/CD pipeline
- Azure App Service deployment
- Prometheus + Grafana monitoring
- Email alerting via Alertmanager

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, Recharts |
| Backend | FastAPI, Python 3.11, Uvicorn |
| Database | PostgreSQL, Redis |
| ML | scikit-learn, statsmodels, pandas |
| RAG | ChromaDB, sentence-transformers, PyMuPDF |
| LLM | Google Gemini API |
| Auth | JWT, Bcrypt |
| DevOps | Docker, GitHub Actions, Azure |
| Monitoring | Prometheus, Grafana, Alertmanager |
| Orchestration | Apache Airflow |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         React Frontend (5173)        │
└────────────────┬────────────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
┌─────▼──────┐      ┌───────▼──────┐
│  FastAPI   │      │  RAG Pipeline │
│ (Port 8000)│      │  (Port 8001)  │
│            │      │               │
│ • Auth     │      │ • PDF Extract │
│ • Invoices │      │ • Embeddings  │
│ • ML APIs  │      │ • ChromaDB    │
│ • Reports  │      │ • Gemini LLM  │
└─────┬──────┘      └───────┬───────┘
      │                     │
┌─────▼──────┐      ┌───────▼───────┐
│ PostgreSQL │      │   ChromaDB    │
└────────────┘      └───────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+

### 1. Clone Repository
```bash
git clone https://github.com/randeepraj2003/ledger-ai.git
cd ledger-ai
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Setup
```bash
cp .env.example .env
# Fill in your actual values in .env
```

### 4. Run Backend
```bash
# Terminal 1 — Main Backend
uvicorn main:app --port 8000

# Terminal 2 — RAG Pipeline
uvicorn rag_pipeline:app --port 8001
```

### 5. Run Frontend
```bash
# Terminal 3
cd frontend
npm install
npm run dev
```

### 6. Open App

```
```

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in:

Get free Gemini API key at: https://aistudio.google.com/apikey

---

## 📁 Project Structure

```
ledger-ai/
├── .github/
│   └── workflows/
│       └── ci_cd.yml          # GitHub Actions CI/CD
├── azure/
│   ├── azure_deploy.sh        # Azure deployment script
│   └── startup.sh
├── monitoring/
│   ├── prometheus.yml         # Prometheus config
│   ├── alert_rules.yml        # Alert rules
│   ├── alertmanager.yml       # Email alerts
│   └── docker-compose.monitoring.yml
├── backend/
│   ├── airflow/
│   │   └── dags/
│   │       └── ledgerai_pipeline.py  # Airflow DAG
│   ├── app/
│   │   ├── routers/           # API endpoints
│   │   ├── models/            # DB models
│   │   └── config.py
│   ├── main.py                # FastAPI entry
│   ├── rag_pipeline.py        # RAG + Gemini
│   ├── metrics.py             # Prometheus metrics
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── DashboardPage.jsx
│       │   ├── InvoicesPage.jsx
│       │   └── ChatPage.jsx
│       └── services/
│           └── api.js
├── docker-compose.yml
└── README.md
```

---

## 📊 ML Models Detail

| Model | Purpose | Result |
|-------|---------|--------|
| ARIMA | Spending forecast | 3-month prediction |
| Random Forest | Amount prediction | MAE: $407 |
| Isolation Forest | Anomaly detection | Risk: 0-100% |
| DBSCAN | Pattern clustering | 2 clusters + outliers |
| K-Means | Vendor risk | HIGH/MEDIUM/LOW |
| TF-IDF | Duplicate detection | Cosine similarity |

---

## 🔄 Airflow Pipeline

Runs automatically every midnight:
```
Fetch Invoices → Anomaly Detection → Spending Forecast
             → Vendor Risk Scoring → Daily Report
             → Save to DB → Email Alert
```

---

## ☁️ Cloud Deployment

Deploy to Azure App Service:
```bash
cd azure
bash azure_deploy.sh
```

---

## 📈 Monitoring

Start monitoring stack (requires Docker):
```bash
cd monitoring
docker compose -f docker-compose.monitoring.yml up -d
```

Access:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Alerts: http://localhost:9093

---

## 👤 Author

**Randeep Raj**
- GitHub: [@randeepraj2003](https://github.com/randeepraj2003)
- LinkedIn: [linkedin.com/in/randeep-raj](https://linkedin.com/in/randeep-raj)
- Email: Randeepraj207107@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ If you found this helpful, please give it a star!
