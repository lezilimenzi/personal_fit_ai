"""BGE-M3 모델로 텍스트를 Dense + Sparse 벡터로 인코딩."""

from FlagEmbedding import BGEM3FlagModel


class Embedder:
    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode(self, texts: list[str]) -> tuple:
        """
        Returns:
            (dense_vecs: ndarray (N, 1024), lexical_weights: list[dict])
        """
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"], output["lexical_weights"]
