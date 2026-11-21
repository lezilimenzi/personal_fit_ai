"""FAISS 인덱스 + Sparse weights 관리."""

import json
import os
import pickle

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.sparse_weights: list[dict] = []
        self.doc_meta: list[dict] = []

    def add(self, dense_vecs: np.ndarray, sparse_weights: list[dict],
            metadata: list[dict]) -> None:
        # L2 정규화 → Inner Product = 코사인 유사도
        faiss.normalize_L2(dense_vecs)
        self.index.add(dense_vecs)
        self.sparse_weights.extend(sparse_weights)
        self.doc_meta.extend(metadata)

    def search_dense(self, query_vec: np.ndarray, top_n: int) -> tuple:
        """Returns: (scores: ndarray (1, top_n), indices: ndarray (1, top_n))"""
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_n)
        return scores, indices

    def get_sparse_weights(self, idx: int) -> dict:
        if 0 <= idx < len(self.sparse_weights):
            return self.sparse_weights[idx]
        return {}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "dense.index"))
        with open(os.path.join(path, "sparse_weights.pkl"), "wb") as f:
            pickle.dump(self.sparse_weights, f)
        with open(os.path.join(path, "doc_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.doc_meta, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(os.path.join(path, "dense.index"))
        with open(os.path.join(path, "sparse_weights.pkl"), "rb") as f:
            self.sparse_weights = pickle.load(f)
        with open(os.path.join(path, "doc_meta.json"), "r", encoding="utf-8") as f:
            self.doc_meta = json.load(f)
