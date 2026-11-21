"""벡터 인덱스 빌드 스크립트. 문서 로드 → 청킹 → 임베딩 → FAISS 저장."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.rag.document_loader import DocumentLoader
from src.rag.chunker import Chunker
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore

from transformers import AutoTokenizer


def main():
    print("[1/5] 토크나이저 로드...")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL, trust_remote_code=True)

    print("[2/5] 문서 로드...")
    loader = DocumentLoader(
        raw_knowledge_dir=config.RAW_KNOWLEDGE_DIR,
        raw_catalog_dir=config.RAW_CATALOG_DIR,
        recommend_data_dir=config.RECOMMEND_DATA_DIR,
    )
    docs = loader.load_all()
    print(f"  → {len(docs)}개 문서 로드됨")

    loader.save_docs(docs, config.DOCS_DIR)
    print(f"  → {config.DOCS_DIR}에 JSON 저장 완료")

    print("[3/5] 청킹...")
    chunker = Chunker(tokenizer=tokenizer, max_tokens=config.CHUNK_MAX_TOKENS)
    chunks = chunker.chunk_all(docs)
    print(f"  → {len(chunks)}개 청크 생성됨")

    print("[4/5] 임베딩 (BGE-M3)...")
    embedder = Embedder(config.EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    dense_vecs, sparse_weights = embedder.encode(texts)
    print(f"  → Dense: {dense_vecs.shape}, Sparse: {len(sparse_weights)}개")

    print("[5/5] FAISS 인덱스 저장...")
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "title": c["title"],
            "text": c["text"],
            "category": c["category"],
        }
        for c in chunks
    ]
    store = VectorStore(dim=1024)
    store.add(dense_vecs, sparse_weights, metadata)
    store.save(config.VECTORSTORE_DIR)
    print(f"  → {config.VECTORSTORE_DIR}에 인덱스 저장 완료")
    print(f"\n완료! 총 {len(docs)}개 문서 → {len(chunks)}개 청크 인덱싱됨")


if __name__ == "__main__":
    main()
