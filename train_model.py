import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from supabase import create_client

load_dotenv()

# No hardcoded fallback: keys live in .env only
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def fetch_all(table: str, page: int = 1000) -> list:
    rows, start = [], 0
    while True:
        chunk = (supabase.table(table).select("*")
                 .range(start, start + page - 1).execute().data)
        rows += chunk
        if len(chunk) < page:
            return rows
        start += page


print("Fetching course data from Supabase...")
df = pd.DataFrame(fetch_all("course_market_database"))
print(f"Fetched {len(df)} rows")

# Feature mapping (case-insensitive)
scope_mapping = {"great": 100.0, "good": 75.0, "bad": 40.0}
df["scope_score"] = (df["future_scope"].astype(str).str.lower().str.strip()
                     .map(scope_mapping).fillna(50.0))

feature_cols = ["min_annual_tuition_fee_inr", "max_annual_tuition_fee_inr",
                "median_salary_inr", "scope_score"]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows missing key numbers instead of filling 0 (fake zero-fee courses skew KNN)
df = df.dropna(subset=["min_annual_tuition_fee_inr", "median_salary_inr"]).copy()
df["max_annual_tuition_fee_inr"] = df["max_annual_tuition_fee_inr"].fillna(
    df["min_annual_tuition_fee_inr"])
df = df.reset_index(drop=True)  # keeps iloc indices aligned with KNN rows

X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Euclidean (not cosine): cosine ignores magnitude, so ₹50k vs ₹5L fees looked the same
ml_model = NearestNeighbors(n_neighbors=min(15, len(df)), metric="euclidean")
ml_model.fit(X_scaled)

artifacts = {"model": ml_model, "scaler": scaler,
             "courses_df": df, "feature_cols": feature_cols}

model_filename = "prism_ml_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(artifacts, f)

print(f"✅ Saved '{model_filename}' ({os.path.getsize(model_filename)} bytes)")