import os
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

load_dotenv()

# Database Connection Setup
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://uvmfmurehnlkctlvrljx.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV2bWZtdXJlaG5sa2N0bHZybGp4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTEzODc0OTMsImV4cCI6MjEwNjk2MzQ5M30.0fJSDa22DoXYKsbN8qQcSQEAR1Hm42gPgysHKtjtl_c")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 1. Fetch Course Data from Supabase
print("Fetching course data from Supabase...")
courses_res = supabase.table("course_market_database").select("*").execute()
df = pd.DataFrame(courses_res.data)

# 2. Feature Mapping & Numeric Cleaning
scope_mapping = {"Great": 100.0, "Good": 75.0, "Bad": 40.0}
df["scope_score"] = df["future_scope"].map(scope_mapping).fillna(50.0)

feature_cols = ["min_annual_tuition_fee_inr", "max_annual_tuition_fee_inr", "median_salary_inr", "scope_score"]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

X = df[feature_cols].values

# 3. Fit Standard Scaler and KNN ML Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ml_model = NearestNeighbors(n_neighbors=15, metric="cosine")
ml_model.fit(X_scaled)

# 4. Save Artifacts using Pure Binary Pickle
artifacts = {
    "model": ml_model,
    "scaler": scaler,
    "courses_df": df,
    "feature_cols": feature_cols
}

model_filename = "prism_ml_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(artifacts, f)

print(f"✅ SUCCESS: Saved '{model_filename}' ({os.path.getsize(model_filename)} bytes)")