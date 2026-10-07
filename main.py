import os
import pickle
import pandas as pd
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

app = FastAPI(
    title="PRISM Recommendation Engine",
    description="FastAPI Backend for PRISM AI-Driven Student Career & Course Recommendations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


class MLPredictionRequest(BaseModel):
    student_id: Optional[int] = Field(default=1)
    academic_score: Optional[float] = Field(default=85.0, ge=0.0, le=100.0)
    technical_aptitude: Optional[float] = Field(default=80.0, ge=0.0, le=100.0)
    domain_interest: Optional[float] = Field(default=75.0, ge=0.0, le=100.0)
    preferred_budget: Optional[float] = Field(default=50000.0, ge=0.0)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


class CustomMCDARequest(BaseModel):
    weight_academic: Optional[float] = Field(default=0.4)
    weight_aptitude: Optional[float] = Field(default=0.3)
    weight_budget: Optional[float] = Field(default=0.3)
    max_budget: Optional[float] = Field(default=100000.0)
    top_k: Optional[int] = Field(default=5)


ml_model = None
scaler = None
ml_courses_df = None

MODEL_FILENAME = "prism_ml_model.pkl"

def load_artifacts():
    global ml_model, scaler, ml_courses_df

    if not os.path.exists(MODEL_FILENAME):
        print(f"⚠️ Warning: '{MODEL_FILENAME}' missing. Run 'python train_model.py' first.")
        return

    try:
        with open(MODEL_FILENAME, "rb") as f:
            artifacts = pickle.load(f)

        if isinstance(artifacts, dict):
            ml_model = artifacts.get("model")
            scaler = artifacts.get("scaler")
            ml_courses_df = artifacts.get("courses_df")
            print("✅ Model artifacts loaded successfully using pickle.")
    except Exception as e:
        print(f"❌ Unpickling Error: {e}")

load_artifacts()


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/api/recommend/ml")
def predict_knn_recommendations(payload: MLPredictionRequest):
    if ml_model is None or scaler is None:
        return {
            "status": "fallback",
            "message": "Model artifact not loaded.",
            "recommendations": []
        }

    try:
        target_budget = float(payload.preferred_budget)
        target_salary_benchmark = 500000.0
        target_scope = float((payload.technical_aptitude + payload.domain_interest) / 2.0)

        features = np.array([[
            target_budget,
            target_budget,
            target_salary_benchmark,
            target_scope
        ]])

        features_scaled = scaler.transform(features)
        distances, indices = ml_model.kneighbors(features_scaled, n_neighbors=payload.top_k)

        recommendations = []
        if ml_courses_df is not None and not ml_courses_df.empty:
            for idx, dist in zip(indices[0], distances[0]):
                course_info = ml_courses_df.iloc[idx].to_dict()
                clean_info = {k: (None if pd.isna(v) else v) for k, v in course_info.items()}
                clean_info["distance"] = round(float(dist), 4)
                clean_info["confidence_score"] = round(float(1 / (1 + dist)), 4)
                recommendations.append(clean_info)

        return {
            "status": "success",
            "student_id": payload.student_id,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Inference Error: {str(e)}")


@app.post("/api/recommend/custom")
def custom_mcda_recommendations(payload: CustomMCDARequest):
    return {
        "status": "success",
        "method": "MCDA",
        "recommendations": []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)