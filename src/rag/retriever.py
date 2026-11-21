"""Dense + Sparse 점수를 결합하여 후보 문서를 검색."""

import numpy as np


def _compute_lexical_score(query_sparse: dict, doc_sparse: dict) -> float:
    """쿼리와 문서의 sparse weight 간 내적."""
    score = 0.0
    for token_id, q_weight in query_sparse.items():
        if token_id in doc_sparse:
            score += q_weight * doc_sparse[token_id]
    return score


class HybridRetriever:
    def __init__(self, embedder, vector_store, w_dense: float = 0.7,
                 w_sparse: float = 0.3):
        self.embedder = embedder
        self.store = vector_store
        self.w_dense = w_dense
        self.w_sparse = w_sparse

    def retrieve(self, query: str, top_n: int = 20) -> list[dict]:
        q_dense, q_sparse = self.embedder.encode([query])

        dense_scores, dense_ids = self.store.search_dense(q_dense, top_n)

        sparse_scores = []
        for idx in dense_ids[0]:
            if idx < 0:
                sparse_scores.append(0.0)
                continue
            doc_sparse = self.store.get_sparse_weights(idx)
            sparse_scores.append(_compute_lexical_score(q_sparse[0], doc_sparse))

        final_scores = (
            self.w_dense * dense_scores[0]
            + self.w_sparse * np.array(sparse_scores)
        )

        ranked_indices = np.argsort(-final_scores)[:top_n]
        results = []
        for i in ranked_indices:
            idx = dense_ids[0][i]
            if idx < 0:
                continue
            meta = self.store.doc_meta[idx].copy()
            meta["score"] = float(final_scores[i])
            results.append(meta)

        return results
