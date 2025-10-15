# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI(title="Ecommerce Recommender", version="0.1.0")

# ----------------- Pydantic models -----------------
class RecommendRequest(BaseModel):
    user_id: str
    limit: int = 5
    tone: str = "friendly"

# ----------------- Data loading -----------------
products = pd.read_csv("../data/products.csv")  # product_id, title, category, brand, price, attributes
interactions = pd.read_csv("../data/interactions.csv")  # user_id, product_id, event_type, value

# ----------------- Collaborative filtering -----------------
def build_interaction_matrix(interactions: pd.DataFrame):
    df = interactions.copy()
    if 'value' not in df.columns:
        df['value'] = 1.0
    weights = {'view':1.0,'click':2.0,'cart':3.0,'purchase':5.0}
    df['weight'] = df['event_type'].map(weights).fillna(1.0) * df['value']
    pivot = df.pivot_table(index='user_id', columns='product_id', values='weight', aggfunc='sum', fill_value=0.0)
    return pivot

interaction_matrix = build_interaction_matrix(interactions)

def item_similarity_matrix(interaction_matrix: pd.DataFrame):
    if interaction_matrix.empty:
        return pd.DataFrame()
    mat = np.asarray(interaction_matrix.values, dtype=float)
    item_vecs = mat.T
    norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    item_norm = item_vecs / norms
    sim = item_norm @ item_norm.T
    return pd.DataFrame(sim, index=interaction_matrix.columns, columns=interaction_matrix.columns)

sim_matrix = item_similarity_matrix(interaction_matrix)

def score_items_for_user(user_id: str, interaction_matrix: pd.DataFrame, sim_df):
    if interaction_matrix.empty or user_id not in interaction_matrix.index:
        return {}
    user_vec = interaction_matrix.loc[user_id]
    interacted = user_vec[user_vec>0].index.tolist()
    scores = {}
    for item in sim_df.index:
        if item in interacted:
            continue
        sims = sim_df.loc[item, interacted] if interacted else []
        vals = user_vec[interacted] if interacted else []
        if len(sims) == 0:
            scores[item] = 0.0
        else:
            sims_arr = np.asarray(sims, dtype=float)
            vals_arr = np.asarray(vals, dtype=float)
            scores[item] = float(np.dot(sims_arr, vals_arr))
    return scores

# ----------------- Content-based filtering -----------------
# Use title + category + brand as content
products['content'] = products['title'] + " " + products['category'] + " " + products['brand']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(products['content'].fillna(''))

def content_score(user_history: pd.Series):
    if user_history.empty:
        return {}
    
    user_profile = tfidf.transform(products.loc[products['product_id'].isin(user_history), 'content'])
    sims = linear_kernel(np.asarray(user_profile), np.asarray(tfidf_matrix)).flatten()
    scores = {pid: float(score) for pid, score in zip(products['product_id'], sims)}
    return scores

# ----------------- FastAPI routes -----------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    user_id = req.user_id
    limit = req.limit

    # Collaborative filtering
    cf_scores = score_items_for_user(user_id, interaction_matrix, sim_matrix)

    # Content-based filtering
    user_history = interactions.loc[interactions['user_id'] == user_id, 'product_id']
    cb_scores = content_score(user_history)

    # Combine scores (simple sum)
    all_scores = {}
    for pid in products['product_id']:
        all_scores[pid] = cf_scores.get(pid, 0.0) + cb_scores.get(pid, 0.0)

    # Sort and pick top-N
    recommended = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    recommended_products = [products.loc[products['product_id'] == pid, 'title'].values[0] for pid, _ in recommended]

    # Example explanation
    explanation = f"Recommended based on your previous interactions and product similarities in a {req.tone} tone."

    return {
        "user_id": user_id,
        "recommendations": recommended_products,
        "explanation": explanation
    }
