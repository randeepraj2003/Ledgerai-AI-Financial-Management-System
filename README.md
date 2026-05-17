# Ledgerai-Finance-Dashboard
### AI-Powered Invoice & Financial Management System
Where finance meets AI — RAG pipeline, 6 ML models, vector search &amp; LLM-powered chat built on FastAPI + React + PostgreSQL.

##  Project Overview

LedgerAI is a full-stack, AI-powered financial management platform designed to automate invoice processing, detect anomalies, predict spending patterns, and enable intelligent document Q&A using RAG (Retrieval-Augmented Generation). The system combines traditional machine learning models with modern LLM capabilities to provide a comprehensive financial intelligence tool.

#  Problem It Solves

| Problem | LedgerAI Solution |
|---------|------------------|
| Manual invoice review is slow | AI auto-extracts and categorizes PDF invoices |
| Hard to spot fraudulent invoices | Isolation Forest anomaly detection flags suspicious ones |
| No spending forecasting | ARIMA time series predicts future monthly spend |
| Can't ask questions about invoices | RAG pipeline enables natural language Q&A over PDFs |
| Vendor risk is unknown | K-Means clustering scores each vendor's risk level |
| Duplicate invoices go unnoticed | TF-IDF cosine similarity detects duplicates |

---

##  System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FRONTEND                          │
│              React + Vite (Port 5173)                │
│   Dashboard │ Invoices │ AI Assistant │ Analytics    │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP Requests
          ┌───────────┴───────────┐
          │                       │
┌─────────▼─────────┐   ┌────────▼────────┐
│   MAIN BACKEND    │   │  RAG PIPELINE   │
│ FastAPI (Port 8000)│   │FastAPI (Port 8001)│
│                   │   │                  │
│ • Auth (JWT)      │   │ • PDF Extraction │
│ • Invoice CRUD    │   │ • Chunking       │
│ • ML Models       │   │ • Embeddings     │
│ • Analytics       │   │ • Vector Search  │
│ • Reports         │   │ • Gemini LLM     │
└─────────┬─────────┘   └────────┬────────┘
          │                       │
┌─────────▼─────────┐   ┌────────▼────────┐
│   PostgreSQL DB   │   │    ChromaDB      │
│  (Invoice Data)   │   │ (Vector Store)   │
└───────────────────┘   └─────────────────┘
```

---

##  Tech Stack

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18.x | UI framework |
| Vite | 5.x | Build tool & dev server |
| Recharts | 2.x | Charts & data visualization |
| CSS Variables | - | Dark theme styling |

### Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| FastAPI | 0.100+ | REST API framework |
| Python | 3.11 | Core language |
| Uvicorn | 0.20+ | ASGI server |
| Pydantic | 2.x | Data validation |
| SQLAlchemy | 2.x | ORM for database |
| PostgreSQL | 15.x | Primary database |
| Redis | 7.x | Caching & session store |
| JWT (Jose) | - | Authentication tokens |
| Bcrypt | - | Password hashing |

### Machine Learning
| Model | Library | Purpose |
|-------|---------|---------|
| ARIMA | statsmodels | Time series spend forecasting |
| Random Forest Regressor | scikit-learn | Invoice amount prediction |
| DBSCAN | scikit-learn | Spending pattern discovery |
| Isolation Forest | scikit-learn | Anomaly detection |
| K-Means Clustering | scikit-learn | Vendor risk scoring |
| TF-IDF + Cosine Similarity | scikit-learn | Duplicate invoice detection |
| NLP Categorizer | scikit-learn | Auto invoice categorization |

### RAG Pipeline
| Technology | Purpose |
|-----------|---------|
| PyMuPDF (fitz) | PDF text extraction |
| sentence-transformers | Text embeddings (all-MiniLM-L6-v2) |
| ChromaDB | Vector database for similarity search |
| Google Gemini API | LLM for intelligent Q&A (free tier) |
| python-dotenv | Environment variable management |

### DevOps & Tools
| Tool | Purpose |
|------|---------|
| Git & GitHub | Version control |
| Docker | Containerization |
| VS Code | Development environment |
| Postman/Swagger | API testing |
| .env + .gitignore | Secure credential management |

---

##  ML Models — Deep Dive

### 1. ARIMA Time Series Forecast
- **Full Name:** AutoRegressive Integrated Moving Average
- **Purpose:** Predicts future monthly spending based on historical invoice data
- **Output:** 3-month forecast with confidence intervals
- **Fallback:** Average-based prediction when insufficient historical data

### 2. Random Forest Regressor
- **Trees:** 100 estimators
- **Purpose:** Predicts expected invoice amount for each vendor
- **MAE:** ~$407 (Mean Absolute Error)
- **Features Used:** Vendor encoding, Category, Month, Quarter, Tax amount, Status
- **Output:** Predicted amount vs actual + deviation percentage

### 3. DBSCAN Clustering
- **Full Name:** Density-Based Spatial Clustering of Applications with Noise
- **Parameters:** eps=0.8, min_samples=2
- **Features:** Amount, Tax ratio, Month
- **Output:** Spending pattern clusters + outlier detection

### 4. Isolation Forest
- **Type:** Unsupervised anomaly detection
- **Purpose:** Detects statistically unusual invoices
- **Output:** Risk score (0-100%) for each invoice
- **Key Finding:** Flags invoices that deviate from normal patterns

### 5. K-Means Clustering
- **Purpose:** Vendor risk profiling
- **Output:** HIGH / MEDIUM / LOW risk classification
- **Factors:** Spend amount, frequency, anomaly history

### 6. TF-IDF Cosine Similarity
- **Full Name:** Term Frequency-Inverse Document Frequency
- **Purpose:** Detect duplicate or near-duplicate invoices
- **Method:** Converts invoice text to vectors, measures similarity
- **Threshold:** Flags invoices with >90% similarity

---

##  RAG Pipeline — How It Works

```
Step 1: User uploads PDF invoice
           ↓
Step 2: PyMuPDF extracts text from PDF
           ↓
Step 3: Text split into 500-char chunks (with 50-char overlap)
           ↓
Step 4: sentence-transformers converts chunks to embeddings
        (all-MiniLM-L6-v2 model — 384 dimensions)
           ↓
Step 5: Embeddings stored in ChromaDB (persistent vector store)
           ↓
Step 6: User asks a question
           ↓
Step 7: Question embedded using same model
           ↓
Step 8: ChromaDB cosine similarity search → Top 5 relevant chunks
           ↓
Step 9: Chunks + Question sent to Gemini API
           ↓
Step 10: Gemini generates intelligent answer with source citations
           ↓
Step 11: Answer displayed in chat with PDF source badges
```

---

##  Security Implementation

- **Authentication:** JWT (JSON Web Tokens) with 24-hour expiry
- **Password:** Bcrypt hashing (never stored in plain text)
- **API Keys:** Stored in `.env` file, never committed to GitHub
- **CORS:** Configured to allow only frontend origin
- **Environment:** `.gitignore` excludes all sensitive files

---

##  Project Structure

```
finance-dashboard/
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── DashboardPage.jsx    # Main dashboard + ML charts
│   │   │   ├── InvoicesPage.jsx     # Invoice management
│   │   │   └── ChatPage.jsx         # AI Assistant with RAG
│   │   ├── services/
│   │   │   └── api.js               # API service layer
│   │   └── App.jsx
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── routers/
│   │   │   ├── auth.py              # Login, register, JWT
│   │   │   ├── invoices.py          # Invoice CRUD
│   │   │   ├── analytics.py         # ML model endpoints
│   │   │   ├── ai_chat.py           # Chat API
│   │   │   ├── reports.py           # Report generation
│   │   │   └── forecast.py          # ARIMA forecasting
│   │   ├── config.py                # Settings & env vars
│   │   └── database.py              # DB connection
│   ├── rag_pipeline.py              # RAG + Gemini system
│   ├── main.py                      # FastAPI app entry
│   ├── requirements.txt
│   ├── .env                         # ← NOT in GitHub
│   ├── .env.example                 # ← Safe template
│   └── .gitignore
│
└── README.md
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+

### Setup Steps

```bash
# 1. Clone repository
git clone https://github.com/your-username/ledgerai.git
cd ledgerai

# 2. Backend setup
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Environment setup
cp .env.example .env
# Fill in your actual values in .env

# 4. Start main backend
uvicorn main:app --port 8000

# 5. Start RAG pipeline (new terminal)
uvicorn rag_pipeline:app --port 8001

# 6. Frontend setup (new terminal)
cd ../frontend
npm install
npm run dev
```


##  Skills Demonstrated

### Data Science & ML
- Time series forecasting (ARIMA)
- Supervised learning (Random Forest)
- Unsupervised learning (K-Means, DBSCAN)
- Anomaly detection (Isolation Forest)
- NLP & text similarity (TF-IDF)
- Feature engineering & model evaluation

### AI & LLM
- RAG (Retrieval-Augmented Generation) architecture
- Vector embeddings & similarity search
- Prompt engineering for financial domain
- LLM API integration (Google Gemini)
- ChromaDB vector database

### Full Stack Development
- REST API design with FastAPI
- React component architecture
- Database design with PostgreSQL
- JWT authentication flow
- PDF processing & extraction

### DevOps & Best Practices
- Environment variable management
- Git version control
- Docker containerization
- API documentation (Swagger)
- Security best practices

---

*Built  by Randeep Raj | LedgerAI v1.0 | May 2026*
