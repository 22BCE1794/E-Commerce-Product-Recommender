import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv
from app.recommender.item_cf import build_interaction_matrix, item_similarity_matrix, score_items_for_user
from recommender.content_based import ContentScorer
from llm import generate_explanation, LLM_MOCK

# Load environment variables
load_dotenv()

# -------------------------------
# Paths
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
INTERACTIONS_CSV = os.path.join(DATA_DIR, "interactions.csv")

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Ecommerce Recommender")

class RecommendRequest(BaseModel):
    user_id: str
    limit: int = 5
    tone: str = "friendly"

# -------------------------------
# Load data
# -------------------------------
try:
    print(f"📂 Loading data from: {DATA_DIR}")
    products_df = pd.read_csv(PRODUCTS_CSV, converters={'attributes': lambda x: eval(x) if pd.notna(x) else {}})
    interactions_df = pd.read_csv(INTERACTIONS_CSV)
    print(f"✅ Loaded {len(products_df)} products, {len(interactions_df)} interactions")
except Exception as e:
    print("❌ Error loading CSV files:", e)
    raise

# -------------------------------
# Build recommender artifacts
# -------------------------------
try:
    interaction_matrix = build_interaction_matrix(interactions_df)
    sim_df = item_similarity_matrix(interaction_matrix)
    content_scorer = ContentScorer(products_df)
    print("✅ Recommender initialized successfully")
except Exception as e:
    print("❌ Error initializing recommender:", e)
    raise

# -------------------------------
# Routes
# -------------------------------
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "llm_mock": bool(LLM_MOCK)}

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    """Generate recommendations for a user"""
    try:
        user_id = req.user_id
        if user_id not in interaction_matrix.index:
            raise HTTPException(status_code=404, detail="user not found or no interactions")

        # Collaborative filtering
        cf_scores = score_items_for_user(user_id, interaction_matrix, sim_df)

        # Get user’s last 10 interactions
        user_history = list(
            interactions_df[interactions_df["user_id"] == user_id]
            .sort_values("timestamp", ascending=False)["product_id"]
            .head(10)
            .values
        )

        # Content-based
        cb_scores = content_scorer.score(user_history)

        # Normalize and combine
        def norm(d):
            if not d:
                return {}
            v = pd.Series(d)
            if v.max() == v.min():
                return {k: 1.0 for k in d}
            nv = (v - v.min()) / (v.max() - v.min())
            return nv.to_dict()

        n_cf = norm(cf_scores)
        n_cb = norm(cb_scores)
        candidates = set(list(n_cf.keys()) + list(n_cb.keys()))
        combined = {c: n_cf.get(c, 0) * 0.6 + n_cb.get(c, 0) * 0.4 for c in candidates}

        # Exclude already seen items
        seen = set(interactions_df[interactions_df["user_id"] == user_id]["product_id"].unique())
        final = {k: v for k, v in combined.items() if k not in seen}

        # Rank and select top-N
        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)[:req.limit]

        # Generate explanations
        out = []
        for pid, score in ranked:
            prod = products_df[products_df["product_id"] == pid].iloc[0].to_dict()
            reason = generate_explanation(user_history, prod, {"score": score}, tone=req.tone)
            out.append({"product": prod, "score": float(score), "explanation": reason})

        return {"user_id": user_id, "recommendations": out}

    except Exception as e:
        print("🔥 ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
