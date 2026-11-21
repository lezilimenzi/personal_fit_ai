import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- 모델 ---
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# --- vLLM ---
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")

# --- 검색 ---
RETRIEVAL_TOP_N = 20
RERANK_TOP_K = 4  # 토큰 한도 내 검색 문서 수 (5→4로 축소)
W_DENSE = 0.7
W_SPARSE = 0.3

# --- 청킹 ---
CHUNK_MAX_TOKENS = 1000

# --- LLM 생성 ---
MAX_MODEL_LEN = 4096
TEMPERATURE = 0.4
TOP_P = 0.8
REPETITION_PENALTY = 1.05
MAX_TOKENS = 512  # 출력 토큰. 전신 등 장문 질의 시 입력 여유 확보 (입력 최대 = 4096-512 = 3584)

# --- 검증 ---
MAX_VALIDATION_RETRIES = 3

# --- 경로 ---
RAW_CATALOG_DIR = str(BASE_DIR / "data" / "raw" / "catalog")
RAW_KNOWLEDGE_DIR = str(BASE_DIR / "data" / "raw" / "knowledge")
RAW_USER_DIR = str(BASE_DIR / "data" / "raw" / "user")
RECOMMEND_DATA_DIR = str(BASE_DIR / "recommend_data")
DOCS_DIR = str(BASE_DIR / "data" / "docs")
VECTORSTORE_DIR = str(BASE_DIR / "vectorstore")
