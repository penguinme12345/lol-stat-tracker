# LoL Coach Tracker – Desktop Application PRD
Version: 1.0  
Author: Jay Patel  
Date: 2026  

---

# 1. Overview

## 1.1 Product Summary

The LoL Coach Tracker Desktop Application transforms the existing CLI-based machine learning analytics pipeline into a cross-platform desktop application using:

- Electron (Frontend Desktop UI)
- FastAPI (Python Backend API)
- Random Forest ML Model (Win Prediction & Feature Importance)
- Local File Storage (User-specific match history & reports)

The application enables users to:

- Sync match history from the Riot API
- Automatically build datasets
- Train a win prediction model
- Generate coaching reports
- View performance trends and improvement targets

This conversion elevates the project from a developer CLI tool into a full-stack desktop analytics product.

---

# 2. Goals & Objectives

## 2.1 Primary Goals

- Convert CLI tool into a fully functional desktop app
- Maintain modular Python ML engine
- Create intuitive UI for match insights
- Enable one-click sync and report generation
- Demonstrate production-ready architecture (Electron + FastAPI)

## 2.2 Success Criteria

- User can install and launch desktop app
- User can connect Riot account
- User can sync matches with one click
- Reports are generated and rendered visually
- ML predictions and feature drivers are displayed clearly
- App runs fully locally without cloud infrastructure

---

# 3. Target Users

- Ranked League of Legends players
- Competitive players seeking performance analytics
- Developers exploring ML-based gaming insights
- Recruiters evaluating system architecture capability

---

# 4. System Architecture

## 4.1 High-Level Architecture

Electron UI (Frontend)  
        ↓ HTTP (localhost)  
FastAPI Backend (Python)  
        ↓  
ML Pipeline (Existing Modules)  
        ↓  
Local Storage (CSV, JSON, Models, Reports)

---

## 4.2 Components

### 4.2.1 Electron (Frontend)

Responsibilities:
- User interface
- Riot credentials input
- Sync and train controls
- Render charts and reports
- Handle loading states and error messaging

Technology:
- Electron
- Node.js
- React (optional)
- Chart.js or Recharts

---

### 4.2.2 FastAPI (Backend)

Responsibilities:
- Expose REST endpoints
- Trigger ingestion
- Build dataset
- Train model
- Generate reports
- Return structured JSON responses

Proposed Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /sync | Download new matches |
| POST | /build | Build dataset |
| POST | /train | Train model |
| GET | /last-game | Return last game report |
| GET | /weekly | Return weekly summary |
| GET | /metrics | Return model metrics |
| GET | /health | Backend health check |

---

### 4.2.3 ML Engine (Existing Modules)

Reused modules:
- ingest.py
- features.py
- train.py
- insights.py

Minimal refactoring:
- Replace CLI calls with service functions
- Return JSON instead of printing output
- Centralize error handling

---

# 5. Functional Requirements

## 5.1 User Onboarding

User enters:
- Riot Game Name
- Tagline
- API Key

App stores API key locally.
API connectivity validated before enabling sync.

---

## 5.2 Sync Matches

When user clicks “Sync Matches”:
- Call /sync endpoint
- Backend fetches new match IDs
- Download raw JSON files
- Update manifest
- Return number of matches downloaded

UI displays:
- Progress indicator
- Success confirmation
- Error messaging if rate limited

---

## 5.3 Build Dataset

- Automatically triggered after sync
- Converts raw JSON → processed CSV
- Validates extracted rows

---

## 5.4 Train Model

- Triggered automatically or manually
- Saves trained model locally
- Returns:
  - Accuracy
  - ROC AUC
  - Top feature importances

---

## 5.5 Last Game Report View

UI displays:

- Match ID
- Predicted win probability
- Win/Loss result
- Champion and role
- Key stats (KDA, CS/min, Damage/min, Gold/min, Vision/min)
- Percentile comparisons
- Top win-rate drivers
- Improvement targets
- Focus goal for next match

---

## 5.6 Weekly Summary View

UI displays:

- Weekly win rate trend (chart)
- Champion pool performance table
- Top win-rate drivers
- Identified performance leaks

---

# 6. Non-Functional Requirements

## 6.1 Performance

- Sync up to 100 matches in under 30 seconds
- Report generation under 2 seconds
- UI remains responsive during API calls

## 6.2 Reliability

- Handle Riot rate limits (HTTP 429)
- Retry on 5xx errors
- Prevent duplicate match downloads

## 6.3 Security

- API key stored locally only
- No external data transmission
- All analytics processed on device

## 6.4 Portability

- Windows (Primary target)
- macOS (Secondary)
- Linux (Optional)

---

# 7. Data Storage Structure

/data  
    /raw/matches  
    manifest.json  
    /processed  
        matches.csv  
/models  
    win_rf.pkl  
    win_rf_metrics.json  
/reports  
    last_game_report.md  
    weekly_summary.md  

Electron renders reports directly from backend JSON responses.

---

# 8. Technical Stack

Backend:
- Python 3.12+
- FastAPI
- Uvicorn
- Pandas
- scikit-learn
- joblib
- requests

Frontend:
- Electron
- Node.js
- React (optional)
- Chart.js / Recharts

---

# 9. Milestones

Phase 1 – Convert CLI to FastAPI  
Phase 2 – Create Electron shell window  
Phase 3 – Connect Electron to FastAPI  
Phase 4 – Render reports visually  
Phase 5 – Add charts and UX polish  
Phase 6 – Package and build installable .exe  

---

# 10. Future Enhancements

- SHAP-based feature attribution (per-match explanations)
- Role-specific ML models
- Pre-game win probability estimator
- Matchup analysis
- Optional cloud sync

---

# 11. Resume Impact Statement

This project demonstrates:

- End-to-end ML pipeline development
- REST API integration with retry logic
- Feature engineering on structured JSON
- Time-series train/test split strategy
- Model persistence and evaluation
- Desktop application architecture (Electron + FastAPI)
- Cross-language system integration (Node + Python)

---

# 12. Risks

| Risk | Mitigation |
|------|------------|
| Riot rate limiting | Backoff + progress UI |
| Electron complexity | Ship MVP first |
| API key misuse | Store locally only |
| Overengineering | Maintain modular backend |

---

# 13. Definition of Done

- Desktop app launches successfully
- User can sync matches
- Model trains successfully
- Reports render visually
- No CLI interaction required
- Fully local execution

