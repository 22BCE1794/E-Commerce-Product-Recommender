from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.recommender import HybridRecommender
from app.llm import LLMWrapper

router = APIRouter()
class RecommendRequest(BaseModel):
    user_id: str
    limit: Optional[int] = 5

class Recommendation(BaseModel):
    product_id: str
    score: float
    reason: str

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[Recommendation]

recommender = HybridRecommender(products_csv='data/products.csv', interactions_csv='data/interactions.csv')
llm = LLMWrapper()

@router.post('/recommend', response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    recs = recommender.recommend(user_id=req.user_id, limit=req.limit)
    out = []
    for pid, score, signals in recs:
        reason = llm.explain(user_id=req.user_id, product_id=pid, signals=signals)
        out.append(Recommendation(product_id=pid, score=score, reason=reason))
    return RecommendResponse(user_id=req.user_id, recommendations=out)
