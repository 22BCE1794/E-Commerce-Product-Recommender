# content_based.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def build_product_text(df: pd.DataFrame):
    """
    Combine product fields into a single text string for TF-IDF.
    Uses title, category, brand, and attributes.
    """
    def row_text(r):
        parts = [str(r.get('title', '')), str(r.get('category', '') or ''), str(r.get('brand', '') or '')]
        attrs = r.get('attributes') or {}
        if isinstance(attrs, dict):
            parts.append(' '.join(f"{k} {v}" for k, v in attrs.items()))
        else:
            parts.append(str(attrs))
        return ' '.join(parts)
    
    df = df.copy()
    df['__text__'] = df.apply(row_text, axis=1)
    return df

class ContentScorer:
    """
    Content-based scorer using TF-IDF and cosine similarity.
    """
    def __init__(self, products_df: pd.DataFrame):
        # Build combined text features
        self.products_df = build_product_text(products_df)
        
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        self.tfidf = self.vectorizer.fit_transform(self.products_df['__text__'])
        
        # Map product_id to TF-IDF row index
        self.index = {pid: idx for idx, pid in enumerate(self.products_df['product_id'].values)}
    def score(self, user_history):
            if not user_history:
                return {}
    
            idxs = [self.index[p] for p in user_history if p in self.index]
            if not idxs:
                return {}
    
            # Convert np.matrix to ndarray
            profile = np.asarray(self.tfidf[idxs].mean(axis=0))
            tfidf_matrix = self.tfidf.toarray()
    
            sims = linear_kernel(profile, tfidf_matrix).flatten()
    
            return {self.products_df.iloc[i]['product_id']: float(sims[i]) for i in range(len(sims))}
