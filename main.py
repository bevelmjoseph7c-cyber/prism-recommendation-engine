import os
import re
import pickle
import traceback
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from supabase import create_client

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = "3.4"
print(f"🚀 PRISM main.py v{VERSION} loaded from {os.path.abspath(__file__)}")

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
COURSE_TABLE = "course_market_database"
STUDENT_TABLE = os.getenv("STUDENT_TABLE", "student_database")  # adjust to your real table
MODEL_PATH = os.path.join(BASE_DIR, "prism_ml_model.pkl")

app = FastAPI(title="PRISM Recommendation Engine", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def all_errors(request, exc):
    traceback.print_exc()
    return JSONResponse(status_code=500,
                        content={"detail": f"{type(exc).__name__}: {exc}"})


# ---------- Domains ----------
DOMAINS = ["tech", "core", "health", "biz", "creative", "science"]
DOMAIN_LABELS = {
    "tech": "Technology & AI", "core": "Core Engineering", "health": "Health & Medicine",
    "biz": "Business & Finance", "creative": "Design, Arts & Media",
    "science": "Science & Research", "other": "Other",
}

# Checked in this order; first match wins
DOMAIN_RULES = [
    ("health", r"mbbs|medic|pharm|nurs|dental|bds|physio|health|ayush|veterin"),
    ("tech", r"computer|software|data|artificial|machine learning|information tech|cyber|\bai\b|\bit\b|\bbca\b|\bml\b|\bcs\b"),
    ("core", r"mechanical|civil|electrical|electronic|engineering|\beng\b|mechatronic|robot|aerospace|chemical|architect|automobile|b\.?tech|\bb\.?e\b"),
    ("biz", r"commerce|bba|mba|management|financ|account|economic|business|b\.?com|marketing|hotel"),
    ("creative", r"design|\barts?\b|\blaw\b|llb|journalism|media|psycholog|literature|fashion|animation|film|humanit|education|\bba\b"),
    ("science", r"b\.?sc|science|research|agricultur|biotech|physics|chemistry|mathemat|statistic|environment|food"),
]


def classify_domain(name: str) -> str:
    n = str(name).lower()
    for dom, pat in DOMAIN_RULES:
        if re.search(pat, n):
            return dom
    return "other"


# ---------- Subjects ----------
SUBJECTS = ["maths", "physics", "chemistry", "economics", "cs", "biology", "humanities"]
SUBJECT_LABELS = {"maths": "Maths", "physics": "Physics", "chemistry": "Chemistry",
                  "economics": "Economics", "cs": "Computer Science",
                  "biology": "Biology", "humanities": "Humanities"}

# Which subjects each course depends on (my estimates, edit freely). First match wins.
SUBJECT_RULES = [
    (r"environment.*eng|eng.*environment", {"chemistry": .35, "biology": .20, "physics": .20, "maths": .25}),
    (r"chemical", {"chemistry": .50, "maths": .25, "physics": .25}),
    (r"robot", {"cs": .40, "physics": .30, "maths": .30}),
    (r"cyber", {"cs": .65, "maths": .35}),
    (r"data", {"cs": .45, "maths": .40, "economics": .15}),
    (r"\bai\b|\bml\b|artificial|machine", {"cs": .50, "maths": .40, "physics": .10}),
    (r"\bcs\b|computer|software|information", {"cs": .70, "maths": .30}),
    (r"mechanical", {"physics": .45, "maths": .40, "chemistry": .15}),
    (r"civil", {"physics": .40, "maths": .40, "chemistry": .10, "economics": .10}),
    (r"electrical|electronic", {"physics": .45, "maths": .45, "cs": .10}),
    (r"architect", {"maths": .35, "physics": .30, "humanities": .35}),
    (r"environment", {"biology": .40, "chemistry": .40, "maths": .10, "humanities": .10}),
    (r"\bphy\b|physics", {"physics": .60, "maths": .40}),
    (r"\bche\b|chemistry", {"chemistry": .80, "maths": .20}),
    (r"\bbio\b|biology|biotech|medic|mbbs|pharm|nurs", {"biology": .60, "chemistry": .40}),
    (r"math|statistic", {"maths": .80, "cs": .10, "economics": .10}),
    (r"econom|financ|commerce|bba|account|business|management", {"economics": .70, "maths": .30}),
    (r"\bfine\b|\barts?\b|\blaw\b|literature|history|humanit|psycholog|journalism|education", {"humanities": 1.0}),
    (r"design", {"humanities": .70, "cs": .15, "maths": .15}),
    (r"media|film|animation|fashion", {"humanities": .80, "economics": .10, "cs": .10}),
]
DOMAIN_SUBJECT_FALLBACK = {
    "tech": {"cs": .5, "maths": .5},
    "core": {"physics": .4, "maths": .4, "chemistry": .2},
    "health": {"biology": .6, "chemistry": .4},
    "biz": {"economics": .6, "maths": .4},
    "creative": {"humanities": 1.0},
    "science": {"maths": .25, "physics": .25, "chemistry": .25, "biology": .25},
}


def course_weights(name: str, domain: str) -> dict:
    n = str(name).lower()
    for pat, w in SUBJECT_RULES:
        if re.search(pat, n):
            return w
    return DOMAIN_SUBJECT_FALLBACK.get(domain, {s: 1.0 for s in SUBJECTS})


# ---------- Data loading ----------
def fetch_all(table: str, page: int = 1000) -> list:
    rows, start = [], 0
    while True:
        chunk = (supabase.table(table).select("*")
                 .range(start, start + page - 1).execute().data)
        rows += chunk
        if len(chunk) < page:
            return rows
        start += page


SCOPE_MAP = {"great": 100.0, "good": 75.0, "bad": 40.0}
courses_df: Optional[pd.DataFrame] = None
load_error: Optional[str] = None


def find_col(df, exact, contains=()):
    lower = {c.lower(): c for c in df.columns}
    if exact.lower() in lower:
        return lower[exact.lower()]
    for c in df.columns:
        if any(k in c.lower() for k in contains):
            return c
    return None


def load_courses():
    global courses_df, load_error
    try:
        df = pd.DataFrame(fetch_all(COURSE_TABLE))
        if df.empty:
            raise ValueError(f"Table '{COURSE_TABLE}' returned 0 rows (check RLS / table name)")
        print("Columns found:", list(df.columns))

        wanted = {
            "min_annual_tuition_fee_inr": ("min_annual_tuition_fee_inr", ("min", "fee")),
            "max_annual_tuition_fee_inr": ("max_annual_tuition_fee_inr", ("max", "fee")),
            "median_salary_inr": ("median_salary_inr", ("salary",)),
            "future_scope": ("future_scope", ("scope",)),
            "course_name": ("course_name", ("course", "program", "name")),
            "state_name": ("state_name", ("state",)),
        }
        for target, (exact, kw) in wanted.items():
            found = find_col(df, exact, kw)
            if found and found != target:
                df[target] = df[found]
            elif not found:
                print(f"⚠️ No column for '{target}', using defaults")
                df[target] = np.nan if target != "future_scope" else "Good"

        for c in ["min_annual_tuition_fee_inr", "max_annual_tuition_fee_inr", "median_salary_inr"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["min_annual_tuition_fee_inr", "median_salary_inr"]).copy()
        if df.empty:
            raise ValueError("All rows lost after cleaning: fee/salary columns are not numeric")

        df["max_annual_tuition_fee_inr"] = df["max_annual_tuition_fee_inr"].fillna(
            df["min_annual_tuition_fee_inr"])
        df["scope_score"] = (df["future_scope"].astype(str).str.lower().str.strip()
                             .map(SCOPE_MAP).fillna(50.0))
        df["salary_pct"] = df["median_salary_inr"].rank(pct=True) * 100

        # Domain + course risk (low scope and high fee-to-salary ratio = riskier)
        df["domain"] = df["course_name"].map(classify_domain)
        ratio = df["min_annual_tuition_fee_inr"] / df["median_salary_inr"].clip(lower=1)
        df["course_risk"] = 0.5 * (100 - df["scope_score"]) + 0.5 * ratio.rank(pct=True) * 100

        # Subject weights per course (rows sum to 1)
        M = np.array([[course_weights(n, d).get(s, 0.0) for s in SUBJECTS]
                      for n, d in zip(df["course_name"], df["domain"])], dtype=float)
        M = M / M.sum(axis=1, keepdims=True)
        for i, s in enumerate(SUBJECTS):
            df[f"w_{s}"] = M[:, i]
        df["key_subjects"] = [", ".join(SUBJECT_LABELS[SUBJECTS[j]] for j in np.argsort(-row)[:2] if row[j] > 0)
                              for row in M]

        courses_df = df.reset_index(drop=True)
        load_error = None
        print(f"✅ Loaded {len(courses_df)} courses | domains: {courses_df['domain'].value_counts().to_dict()}")
    except Exception as e:
        load_error = f"{type(e).__name__}: {e}"
        courses_df = None
        print(f"❌ Could not load courses: {load_error}")


load_courses()

ml_model = scaler = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            art = pickle.load(f)
        ml_model, scaler = art["model"], art["scaler"]
        print("✅ KNN artifacts loaded")
    except Exception as e:
        print(f"⚠️ KNN artifacts not loaded: {e}")


def require_courses():
    if courses_df is None:
        load_courses()
    if courses_df is None:
        raise HTTPException(503, f"Course data not loaded: {load_error}")


def active_domains():
    present = set(courses_df["domain"].unique()) if courses_df is not None else set()
    return [d for d in DOMAINS if d in present]


# ---------- Schemas ----------
class ProfileRequest(BaseModel):
    # Student: per-subject answers keyed by SUBJECTS
    academic_percentage: Optional[float] = Field(None, ge=0, le=100)  # fallback if a mark is missing
    marks: Dict[str, float] = Field(default_factory=dict)             # 0..100
    subject_interest: Dict[str, float] = Field(default_factory=dict)  # 0..3
    subject_skill: Dict[str, float] = Field(default_factory=dict)     # 0..3
    preferred_courses: List[str] = Field(default_factory=list)        # student's picks
    priority: str = "balanced"          # balanced | salary | passion | stability
    # Family
    annual_budget_inr: float = Field(250000.0, ge=0)
    loan_capacity_inr: float = Field(0.0, ge=0)
    relocation_state_preference: str = "Karnataka"
    parent_courses: List[str] = Field(default_factory=list)          # parents' picks
    parent_risk: int = Field(3, ge=1, le=5)
    top_k: int = Field(12, ge=1, le=50)


class MLRequest(BaseModel):
    preferred_budget: float = Field(50000.0, ge=0)
    technical_aptitude: float = Field(80.0, ge=0, le=100)
    domain_interest: float = Field(75.0, ge=0, le=100)
    top_k: int = Field(5, ge=1, le=20)


# ---------- Scoring engine ----------
BASE_W = {"academic": 0.18, "interest": 0.17, "skill": 0.12, "financial": 0.18,
          "market": 0.18, "risk": 0.05, "state": 0.05, "parent": 0.07, "pref": 0.20}
PRIORITY_SHIFT = {
    "salary":    {"market": +0.10, "interest": -0.05, "risk": -0.05},
    "passion":   {"interest": +0.10, "market": -0.10},
    "stability": {"risk": +0.10, "market": -0.05, "interest": -0.05},
}


def get_weights(priority: str, has_pref: bool) -> dict:
    w = dict(BASE_W)
    for k, v in PRIORITY_SHIFT.get(priority.lower().strip(), {}).items():
        w[k] = max(0.0, w[k] + v)
    if not has_pref:
        w["pref"] = 0.0
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def conflict_and_swot(df, p, interests, pref_names, parent_names, parent_doms,
                      marks, subj, budget, loan, state):
    nm = df["course_name"].astype(str).str.lower().str.strip()
    vals = list(interests.values())
    spread = max(vals) - min(vals)

    pref_domains = []
    for n in pref_names:
        d = df.loc[nm == n, "domain"]
        if len(d):
            pref_domains.append(d.iloc[0])

    if pref_domains:
        top_domain = Counter(pref_domains).most_common(1)[0][0]
    elif spread >= 10:
        top_domain = max(interests, key=interests.get)
    else:
        top_domain = None

    # Pool of courses that represent what the student is aiming for
    if pref_names:
        pool = df[df["preferred"]]
    elif top_domain:
        pool = df[df["domain"] == top_domain]
    else:
        pool = df

    # Domain conflict: does the student's leaning fall inside what the parents want?
    if parent_doms:
        if pref_domains:
            inside = sum(d in parent_doms for d in pref_domains)
            domain_c = 100 * (1 - inside / len(pref_domains))
        else:
            domain_c = 100 * (1 - max(interests.get(d, 0.0) for d in parent_doms) / max(max(vals), 1.0))
    else:
        domain_c = 0.0

    # Money conflict: share of the target courses the family cannot afford
    money_c = float(100 * (pool["min_annual_tuition_fee_inr"] > budget + loan).mean()) if len(pool) else 0.0

    # Risk conflict: how much riskier the target courses are than the parents' comfort level
    parent_tol = (p.parent_risk - 1) * 25
    risk_c = float(np.clip(pool["course_risk"].mean() - parent_tol, 0, 100)) if len(pool) else 0.0

    conflict = round(0.4 * domain_c + 0.3 * risk_c + 0.3 * money_c, 1)
    label = "Low" if conflict < 30 else "Moderate" if conflict < 60 else "High"
    parts = {"domain": round(domain_c), "risk": round(risk_c), "money": round(money_c)}
    biggest = max(parts, key=parts.get)
    advice = {
        "domain": "The student's leaning and the parents' preferred courses differ. Look at courses that blend both.",
        "risk": "The courses the student leans towards are riskier than the parents are comfortable with. Compare stable options together.",
        "money": "The student's preferred courses are expensive for this budget. Discuss scholarships, loans or other states.",
    }[biggest] if conflict >= 30 else "Student and family are well aligned."

    # SWOT (rule-based)
    S, W_, O, T = [], [], [], []
    tlabel = DOMAIN_LABELS.get(top_domain, "")
    avg_mark = float(np.mean(marks))
    if avg_mark >= 75:
        S.append(f"Strong marks overall (average {avg_mark:.0f})")
    elif avg_mark < 60:
        W_.append(f"Marks are modest overall (average {avg_mark:.0f})")
    best, worst = int(np.argmax(subj)), int(np.argmin(subj))
    if subj[best] >= 70:
        S.append(f"Strongest subject: {SUBJECT_LABELS[SUBJECTS[best]]} ({subj[best]:.0f}/100)")
    if subj[worst] < 45:
        W_.append(f"Weakest subject: {SUBJECT_LABELS[SUBJECTS[worst]]} ({subj[worst]:.0f}/100)")
    if top_domain:
        S.append(f"Leaning towards {tlabel}")
    else:
        W_.append("No clear field preference yet. Explore more courses")

    for n in pref_names[:3]:
        rows = df[nm == n]
        disp, fit = rows["course_name"].iloc[0], float(rows["academic_fit"].mean())
        if fit >= 80:
            S.append(f"Your pick '{disp}' matches your marks well ({fit:.0f})")
        elif fit < 60:
            T.append(f"Your pick '{disp}' is an academic stretch right now ({fit:.0f})")

    afford = float((df["min_annual_tuition_fee_inr"] <= budget + loan).mean())
    if afford >= 0.6:
        S.append("Budget covers most listed courses")
    elif afford < 0.25:
        W_.append("Budget covers few courses")
        T.append("Risk of heavy debt if a costly course is chosen")
    if len(pool):
        sc = pool["scope_score"].mean()
        who = "Preferred courses have" if pref_names else (tlabel + " has" if tlabel else "Listed courses have")
        if sc >= 75:
            O.append(f"{who} strong future scope")
        else:
            T.append(f"{who} only moderate future scope")
    n_state = int((df["state_name"].astype(str).str.lower().str.strip() == state.lower().strip()).sum())
    if n_state:
        O.append(f"{n_state} course options listed in {state}")
    else:
        T.append(f"No listed courses in {state}")
    if conflict >= 60:
        T.append("High parent-student conflict. Talk through the breakdown")
    elif conflict < 30:
        S.append("Student and family are well aligned")
    if risk_c >= 30:
        T.append("Student's target courses look riskier than the parents' comfort level")
    if pref_names and parent_names and set(pref_names) & set(parent_names):
        S.append("Student and parents both picked at least one same course")

    # Label for what the parents want (their course picks only)
    if parent_names:
        pname = ", ".join(df.loc[nm == n, "course_name"].iloc[0] for n in parent_names[:3])
    else:
        pname = "No preference"

    return {
        "top_domain": tlabel or "No clear preference",
        "parent_domain": pname,
        "conflict_index": conflict, "conflict_label": label,
        "conflict_breakdown": parts, "advice": advice,
        "swot": {"strengths": S, "weaknesses": W_, "opportunities": O, "threats": T},
    }


def score_profile(p: ProfileRequest):
    df = courses_df.copy()
    budget, loan = p.annual_budget_inr, p.loan_capacity_inr
    fee = df["min_annual_tuition_fee_inr"]

    # --- Student subject vectors ---
    fallback = p.academic_percentage if p.academic_percentage is not None else 60.0
    marks = np.array([np.clip(p.marks.get(s, fallback), 0, 100) for s in SUBJECTS], dtype=float)
    inter = np.array([np.clip(p.subject_interest.get(s, 1.5), 0, 3) for s in SUBJECTS], dtype=float)
    skill = np.array([np.clip(p.subject_skill.get(s, 1.5), 0, 3) for s in SUBJECTS], dtype=float)
    subj = 100 * (0.4 * marks / 100 + 0.3 * skill / 3 + 0.3 * inter / 3)   # per-subject score

    Wm = df[[f"w_{s}" for s in SUBJECTS]].values
    course_mark = Wm @ marks
    df["interest_fit"] = (Wm @ inter) / 3 * 100
    df["skill_fit"] = (Wm @ skill) / 3 * 100

    # Academic fit: course-weighted marks vs a difficulty proxy (higher pay = more competitive)
    required = 50 + 40 * df["salary_pct"].values / 100
    df["academic_fit"] = np.where(course_mark >= required, 100.0,
                                  np.clip(100 - 4 * (required - course_mark), 0, 100))

    # Financial constraint solver: budget -> loan -> unaffordable
    gap = fee - budget
    within_loan = 100 - 50 * np.clip(gap, 0, None) / max(loan, 1)
    beyond = np.clip(50 - 100 * (gap - loan) / fee.clip(lower=1), 0, None)
    df["financial_affordability"] = np.where(
        gap <= 0, 100.0, np.where(gap <= loan, within_loan, beyond))

    # Market strength
    df["market_score"] = 0.5 * df["scope_score"] + 0.5 * df["salary_pct"]

    # Risk fit: penalise courses riskier than the parents' comfort level
    tolerance = (p.parent_risk - 1) * 25
    df["risk_fit"] = np.clip(100 - np.clip(df["course_risk"] - tolerance, 0, None), 0, 100)

    # Course preferences (student + parents)
    names_lc = df["course_name"].astype(str).str.lower().str.strip()
    existing = set(names_lc)
    pref_names = sorted({c.lower().strip() for c in p.preferred_courses} & existing)
    parent_names = sorted({c.lower().strip() for c in p.parent_courses} & existing)
    df["preferred"] = names_lc.isin(pref_names)
    df["parent_pick"] = names_lc.isin(parent_names)
    has_pref = bool(pref_names)
    df["pref_fit"] = np.where(df["preferred"], 100.0, 0.0)

    # Domain interests derived from the subject answers
    interests = {d: float(df.loc[df["domain"] == d, "interest_fit"].mean()) for d in active_domains()}

    # Parent domains come only from the parents' course picks
    parent_doms = set(df.loc[df["parent_pick"], "domain"])

    # Parent alignment: exact pick 100, same field 70, otherwise 40. No picks = no constraint.
    if parent_names:
        pa = np.where(df["domain"].isin(parent_doms), 70.0, 40.0)
        df["parent_alignment"] = np.where(df["parent_pick"], 100.0, pa)
    else:
        df["parent_alignment"] = 100.0

    # Geography
    df["state_match"] = (df["state_name"].astype(str).str.lower().str.strip()
                         == p.relocation_state_preference.lower().strip())
    state_score = np.where(df["state_match"], 100.0, 0.0)

    w = get_weights(p.priority, has_pref)
    df["prism_match_score"] = (w["academic"] * df["academic_fit"]
                               + w["interest"] * df["interest_fit"]
                               + w["skill"] * df["skill_fit"]
                               + w["financial"] * df["financial_affordability"]
                               + w["market"] * df["market_score"]
                               + w["risk"] * df["risk_fit"]
                               + w["state"] * state_score
                               + w["parent"] * df["parent_alignment"]
                               + w["pref"] * df["pref_fit"]).round(1)

    top = df.sort_values("prism_match_score", ascending=False).head(p.top_k)
    out = pd.DataFrame({
        "course_name": top["course_name"],
        "state_name": top["state_name"],
        "state_match": top["state_match"].astype(bool),
        "preferred": top["preferred"].astype(bool),
        "parent_pick": top["parent_pick"].astype(bool),
        "domain": top["domain"].map(lambda d: DOMAIN_LABELS.get(d, "Other")),
        "key_subjects": top["key_subjects"],
        "prism_match_score": top["prism_match_score"],
        "min_fee": top["min_annual_tuition_fee_inr"],
        "max_fee": top["max_annual_tuition_fee_inr"],
        "median_salary": top["median_salary_inr"],
        "future_scope": top["future_scope"],
        "academic_fit": top["academic_fit"].round(0),
        "interest_fit": top["interest_fit"].round(0),
        "skill_fit": top["skill_fit"].round(0),
        "financial_affordability": top["financial_affordability"].round(0),
        "risk_fit": top["risk_fit"].round(0),
        "parent_alignment": top["parent_alignment"].round(0),
    })
    recs = out.astype(object).where(out.notna(), None).to_dict("records")

    profile = conflict_and_swot(df, p, interests, pref_names, parent_names, parent_doms,
                                marks, subj, budget, loan, p.relocation_state_preference)
    profile["weights"] = {k: round(v, 3) for k, v in w.items()}
    profile["subject_scores"] = {SUBJECT_LABELS[s]: round(float(subj[i])) for i, s in enumerate(SUBJECTS)}
    return {"status": "success", "profile": profile, "recommendations": recs}


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home():
    path = os.path.join(BASE_DIR, "templates", "index.html")
    if not os.path.exists(path):
        raise HTTPException(500, f"index.html not found at {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.get("/version")
def version():
    return {"version": VERSION, "file": os.path.abspath(__file__)}


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION,
            "courses_loaded": 0 if courses_df is None else len(courses_df),
            "domains": {} if courses_df is None else courses_df["domain"].value_counts().to_dict(),
            "load_error": load_error, "knn_loaded": ml_model is not None}


@app.get("/api/domains")
def list_domains():
    require_courses()
    counts = courses_df["domain"].value_counts().to_dict()
    return {"domains": [{"key": d, "label": DOMAIN_LABELS[d], "courses": int(counts[d])}
                        for d in active_domains()]}


@app.get("/api/courses")
def list_courses():
    """Distinct course names (used for both the student and the parent course picks)."""
    require_courses()
    d = courses_df.drop_duplicates("course_name")[["course_name", "domain", "key_subjects"]]
    items = [{"name": r.course_name, "domain": DOMAIN_LABELS.get(r.domain, "Other"),
              "key_subjects": r.key_subjects} for r in d.itertuples()]
    items.sort(key=lambda x: (x["domain"], x["name"]))
    return {"courses": items}


@app.get("/debug/columns")
def debug_columns():
    rows = supabase.table(COURSE_TABLE).select("*").limit(1).execute().data
    return {"sample_row": rows[0] if rows else None}


@app.post("/api/recommend/custom")
def recommend_custom(p: ProfileRequest):
    require_courses()
    return score_profile(p)


@app.post("/api/recommend/ml")
def recommend_knn(p: MLRequest):
    require_courses()
    if ml_model is None or scaler is None:
        raise HTTPException(503, "KNN model not loaded. Run train_model.py first")
    scope = (p.technical_aptitude + p.domain_interest) / 2
    x = scaler.transform([[p.preferred_budget, p.preferred_budget, 500000.0, scope]])
    dist, idx = ml_model.kneighbors(x, n_neighbors=p.top_k)
    recs = []
    for i, d in zip(idx[0], dist[0]):
        row = courses_df.iloc[i].to_dict()
        row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        row["distance"] = round(float(d), 4)
        recs.append(row)
    return {"status": "success", "recommendations": recs}


# Declared AFTER the fixed routes so "/custom" and "/ml" aren't swallowed
@app.get("/api/recommend/{student_id}")
def recommend_by_id(student_id: str):
    require_courses()
    res = (supabase.table(STUDENT_TABLE).select("*")
           .eq("student_id", student_id).limit(1).execute())
    if not res.data:
        raise HTTPException(404, f"Student {student_id} not found in '{STUDENT_TABLE}'")
    s = res.data[0]
    p = ProfileRequest(
        academic_percentage=float(s.get("academic_percentage") or 70),
        annual_budget_inr=float(s.get("annual_budget_inr") or 0),
        loan_capacity_inr=float(s.get("loan_capacity_inr") or 0),
        relocation_state_preference=str(s.get("relocation_state_preference") or ""),
    )
    result = score_profile(p)
    result["student_id"] = student_id
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 