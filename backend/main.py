from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator, EmailStr
from typing import List, Optional
import joblib
import numpy as np
import re
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# ---
# AI Lead Intent Score API Backend
# Author: Shubham Kumar | Gmail: shubhamkumar831015@gmail.com | Contact: +91 9508741536 | GitHub: https://github.com/newturk/cleardeal2
# Professional FastAPI Backend for Open Source Use
# - Loads trained ML model
# - Exposes /score endpoint for lead scoring
# - Validates input, applies rule-based re-ranker
# - Stores results in memory (no persistent DB)
# ---

app = FastAPI(title="AI Lead Intent Score API", description="Open-source lead scoring API with ML and rule-based re-ranking.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local dev, allow all. Restrict in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = joblib.load("model/model.pkl")

# In-memory storage for scored leads
scored_leads = []

# ---
# Input schema (adjust fields as per model features)
class LeadInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    duration: int = Field(..., ge=0)
    campaign: int = Field(..., ge=1)
    pdays: int
    previous: int = Field(..., ge=0)
    poutcome: str
    comments: str
    consent: bool
    month: str
    day: int

    @validator('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'month')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v

    @validator('consent')
    def consent_required(cls, v):
        if v is not True:
            raise ValueError('Consent is required')
        return v

# ---
# Rule-based re-ranker logic
KEYWORD_RULES = {
    r'urgent': 10,
    r'not interested': -10,
    r'follow up': 5,
    r'call back': 5,
    r'positive': 7,
    r'negative': -7,
    r'needs more information': 3,
    r'considering': 2,
    r'no response': -5,
    r'email brochure': 2,
    r'wants to speak': 4,
}

def rerank_score(initial: float, comments: str) -> float:
    score = initial
    comments_lower = comments.lower()
    for pattern, delta in KEYWORD_RULES.items():
        if re.search(pattern, comments_lower):
            score += delta
    return float(np.clip(score, 0, 100))

# ---
# API endpoint
@app.post("/score")
def score_lead(lead: LeadInput):
    # Prepare input for model (as DataFrame with correct columns)
    input_dict = {
        "age": [lead.age],
        "job": [lead.job],
        "marital": [lead.marital],
        "education": [lead.education],
        "default": [lead.default],
        "balance": [lead.balance],
        "housing": [lead.housing],
        "loan": [lead.loan],
        "contact": [lead.contact],
        "duration": [lead.duration],
        "campaign": [lead.campaign],
        "pdays": [lead.pdays],
        "previous": [lead.previous],
        "poutcome": [lead.poutcome],
        "comments": [lead.comments],
        "consent": [lead.consent],
        "month": [lead.month],
        "day": [lead.day]
    }
    model_input = pd.DataFrame(input_dict)
    initial_pred = model.predict_proba(model_input)[0][1] * 100
    # Rerank
    reranked = rerank_score(initial_pred, lead.comments)
    # Store in memory
    result = {
        "age": lead.age,
        "job": lead.job,
        "marital": lead.marital,
        "education": lead.education,
        "initial_score": round(initial_pred, 2),
        "reranked_score": round(reranked, 2),
        "comments": lead.comments
    }
    scored_leads.append(result)
    return result

# ---
# Endpoint to get all scored leads (for frontend table)
@app.get("/scored-leads")
def get_scored_leads():
    return scored_leads 