# E-commerce Product Recommender - Full Project

This repo contains a minimal end-to-end recommender with LLM-powered explanations (OpenAI).

## Quick start (local)
1. `cd Ecommerce-Recommender/backend`
2. Copy `.env.example` to `.env` and fill `OPENAI_API_KEY`.
3. `pip install -r requirements.txt`
4. Seed data is already included in `backend/data/` (CSV). No DB required; the app reads CSVs.
5. Run backend: `uvicorn app.main:app --reload --port 8000`
6. In another terminal: `streamlit run ../frontend/streamlit_app.py --server.port 8501`
7. Open http://localhost:8501

