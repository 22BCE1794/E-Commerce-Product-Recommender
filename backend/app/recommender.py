import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

class HybridRecommender:
    def __init__(self, products_csv='data/products.csv', interactions_csv='data/interactions.csv'):
        self.products = pd.read_csv(products_csv)
        self.interactions = pd.read_csv(interactions_csv)
        # prepare content model
        self.products['text'] = (self.products['title'].fillna('') + ' ' +
                                 self.products['category'].fillna('') + ' ' +
                                 self.products['brand'].fillna(''))
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.prod_mat = self.tfidf.fit_transform(self.products['text'])
        self.id_to_idx = {pid: i for i, pid in enumerate(self.products['product_id'])}
        # build simple item-item from interactions (binary)
        if not self.interactions.empty:
            piv = pd.crosstab(self.interactions['user_id'], self.interactions['product_id'])
            self.users = piv.index.tolist()
            self.items = piv.columns.tolist()
            self.user_item = piv.values
            try:
                self.item_sim = cosine_similarity(self.user_item.T)
            except Exception:
                self.item_sim = None
        else:
            self.user_item = None
            self.item_sim = None

    def recommend(self, user_id, limit=5):
        # itemcf scores
        all_pids = list(self.products['product_id'])
        item_scores = {pid:0.0 for pid in all_pids}
        if self.item_sim is not None and user_id in getattr(self, 'users', []):
            uidx = self.users.index(user_id)
            user_vec = self.user_item[uidx]
            scores = user_vec.dot(self.item_sim)
            for i,pid in enumerate(self.items):
                item_scores[pid] = float(scores[self.items.index(pid)]) if pid in self.items else 0.0
        # content scores from simple profile (titles of interacted items)
        user_actions = self.interactions[self.interactions['user_id']==user_id]
        recent_titles = self.products[self.products['product_id'].isin(user_actions['product_id'])]['title'].tolist()
        profile_text = ' '.join(recent_titles) if recent_titles else ''
        content_scores = {}
        if profile_text:
            q = self.tfidf.transform([profile_text])
            sims = linear_kernel(q, self.prod_mat).flatten()
            for i,pid in enumerate(all_pids):
                content_scores[pid] = float(sims[i])
        else:
            for pid in all_pids:
                content_scores[pid] = 0.0
        # normalize & combine
        import numpy as np
        item_arr = np.array([item_scores[pid] for pid in all_pids])
        content_arr = np.array([content_scores[pid] for pid in all_pids])
        def norm(x):
            if x.max() - x.min() == 0:
                return np.zeros_like(x)
            return (x - x.min())/(x.max()-x.min())
        n_item = norm(item_arr)
        n_content = norm(content_arr)
        final = 0.6 * n_item + 0.4 * n_content
        ranked_idx = (-final).argsort()[:limit]
        results = []
        for idx in ranked_idx:
            pid = all_pids[idx]
            score = float(final[idx])
            signals = {'itemcf': float(n_item[idx]), 'content': float(n_content[idx])}
            results.append((pid, score, signals))
        return results
