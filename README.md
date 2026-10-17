# PRISM Recommendation Engine

Career recommendation platform built for DataQuest 3.0 (VIT Chennai).
Recommends courses to students and parents by scoring course data across Indian states.

## Features
- Scores 19 courses across 6 states (114 records) using pandas
- FastAPI backend serving recommendations
- Optional KNN model (scikit-learn) for ML-based recommendations
- Web interface for comparing courses side by side

## Tech stack
- Backend: Python, FastAPI
- Database: Supabase
- Data/ML: pandas, numpy, scikit-learn
- Frontend: HTML, JavaScript, Tailwind CSS

## Files
- main.py: FastAPI app and scoring logic
- train_model.py: trains the KNN model
- index.html: frontend
- requirements.txt: dependencies
- Data Quest 3 Hackathon.pptx: hackathon presentation

## Run locally (Windows PowerShell)
git clone https://github.com/bevelmjoseph7c-cyber/prism-recommendation-engine
cd prism-recommendation-engine
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# create a .env file with SUPABASE_URL and SUPABASE_KEY
python main.py
# open http://localhost:8000

## Status
Hackathon prototype. Scoring weights are heuristic, not validated on real labour-market data.
