"""Cross-encoder로 후보 문서를 재정렬. Bi-encoder 검색 결과를 정밀화."""

from FlagEmbedding import FlagReranker


class Reranker:
    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        if not docs:
            return []

        pairs = [[query, doc["text"]] for doc in docs]
        scores = self.model.compute_score(pairs)

        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
