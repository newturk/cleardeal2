# Project Process & Rationale

**Author:** Shubham Kumar  
**Email:** shubhamkumar831015@gmail.com  
**Contact:** +91 9508741536  
**GitHub:** https://github.com/newturk/cleardeal2

## 1. Project Overview
Develop a web-based dashboard that predicts lead intent (0-100) using a machine learning model and a rule-based LLM re-ranker, served via a FastAPI backend and accessible through a modern frontend UI. The system empowers brokers to prioritize high-intent prospects, aiming for a 2-3x conversion lift.

---

## 2. Directory Structure
```
/
  frontend/   # React/Vue/Next.js SPA for UI
  backend/    # FastAPI backend for scoring API
  model/      # ML model training scripts & artifacts
  data/       # Datasets (CSV)
  README.md   # Project documentation
  process.md  # Architecture & process definition (this file)
```

---

## 3. System Architecture

```mermaid
graph TD
  A[User (Broker)] -->|Input Lead Data| B[Frontend SPA]
  B -->|POST /score| C[FastAPI Backend]
  C -->|Load| D[ML Model (.pkl)]
  C -->|Score & Re-rank| E[Scoring Logic]
  E -->|Response| B
  C -->|Temp Store| F[In-memory Leads List]
```

- **Frontend SPA**: Single-page app for lead input and score display.
- **FastAPI Backend**: Receives lead data, validates, computes ML score, applies rule-based re-ranker, returns results.
- **ML Model**: Trained on curated dataset, predicts initial lead intent.
- **Temporary Storage**: In-memory list for scored leads (no persistent DB).

---

## 4. Process Phases

### Phase 1: Project Setup & Data Engineering
- Initialize repo and directory structure.
- Prepare a realistic, pattern-rich synthetic dataset (`leads_dataset.csv`) with at least 7 relevant features (e.g., Phone, Email, Credit Score, Age Group, Family Background, Income, Lead Source, Interaction History, Product Interest Level, Comments, Consent, Lead Intent).
- Document data generation logic and feature relationships.

### Phase 2: Machine Learning Model Development
- Select and justify a tabular model (e.g., XGBoost, GradientBoosting, Logistic Regression).
- Engineer features, preprocess data (encoding, scaling), train model.
- Save trained model as `.pkl` in `/model`.
- Script: `train_model.py` (handles all steps).

### Phase 3: Backend Development (API & Logic)
- FastAPI app exposes `/score` POST endpoint.
- Input: JSON with all model features, `comments`, and `consent`.
- Validate input (email, credit score, consent, etc.).
- Compute initial score (ML model, scaled 0-100).
- Apply rule-based re-ranker (keyword-based score adjustments on `comments`).
- Cap scores [0, 100].
- Store leads in memory for frontend display.
- Respond with Email, Initial Score, Reranked Score, Comments.

### Phase 4: Frontend Development (User Interface)
- SPA with:
  - Lead Input Form (all features, comments, consent checkbox)
  - Scored Leads Table (Email, Initial Score, Reranked Score, Comments)
  - (Bonus) Table sorting, local storage, score distribution chart
- Responsive, user-friendly design (Tailwind CSS optional)
- Client-side validation, error handling
- Live updates on new scores

### Phase 5: Deployment & Documentation
- Deploy frontend (Netlify/GitHub Pages), backend (Render/Fly.io)
- Ensure public URLs, <300ms API latency
- Write a 3-page PDF report (Solution Overview, Architecture, ML Model, Re-ranker, Compliance, Evaluation, Challenges)
- README with setup, usage, deployment instructions

---

## 5. Data Flow Summary
1. User enters lead data in frontend form (with consent).
2. Frontend sends POST request to `/score` API.
3. Backend validates, computes ML score, applies re-ranker, stores and returns result.
4. Frontend updates scored leads table in real time.

---

## 6. Rationale & Best Practices
- **Simplicity & Speed:** Use interpretable, fast ML models for tabular data.
- **Compliance:** No real PII, mock consent, in-memory storage only.
- **User Experience:** Responsive SPA, clear error handling, live feedback.
- **Deployment:** Free, public platforms for accessibility.
- **Documentation:** Clear, concise, and comprehensive for reviewers.

---

## 7. Next Steps
- Begin with data engineering and project scaffolding as per this roadmap.
- Track progress and update documentation as the project evolves. 