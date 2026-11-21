# PLAN.md — Personal Fit AI 상세 구현 계획

> **상태**: 검토 대기 (구현 전 사용자 승인 필요)
> **근거**: RESEARCH.md §1-§64 기술 분석 결과
> **기존 프로토타입**: `prototype_fit_ai.py` (LightGBM 규칙 기반)

---

## 1. 프로젝트 개요

### 1.1 목표

RAG(Retrieval-Augmented Generation) + LLM 기반 **개인화 피트니스 루틴 추천 시스템** 구축.

사용자 프로필(나이, 성별, BMI, 골격근량, 운동경력, 목표, 부상이력)과 매 접속 시 입력(가용시간, 장비, 어제 실시한 운동)을 운동 지식 문서와 결합하여 안전하고 효과적인 운동 루틴을 JSON 형식으로 생성한다.

### 1.2 기존 프로토타입과의 차이

| 항목 | 기존 (`prototype_fit_ai.py`) | 신규 (RAG + LLM) |
|------|---------------------------|-----------------|
| 추천 방식 | LightGBM 이진 분류 + 규칙 기반 | BGE-M3 검색 + Qwen2.5 생성 |
| 지식 소스 | `exercise.csv` 5개 컬럼 | 운동 가이드 문서 코퍼스 (자세, 호흡법, 주의사항 등) |
| 개인화 | 수치 피처 (경력, BMI) | 자연어 프로필 + 운동 이력을 프롬프트에 반영 |
| 출력 | 운동 ID 리스트 + 규칙 기반 세트/중량 | 구조화된 JSON 루틴 (Pydantic 검증) |
| 안전 장치 | ±5% 로드 클램프 | 입력/출력 가드레일 + 위험 운동 필터 |

---

## 2. 기술 스택 요약

| 역할 | 선택 | 근거 (RESEARCH.md) |
|------|------|-------------------|
| **Embedding** | `BAAI/bge-m3` | 한국어 포함 100+언어, Dense+Sparse 동시 지원, 1024d, MIT (§23-29) |
| **Vector DB** | FAISS `IndexFlatIP(1024)` | 소규모 코퍼스에서 정확도 최대, 확장 시 `IndexIVFPQ` 전환 (§10-16) |
| **Reranker** | `bge-reranker-v2-m3` | BGE-M3와 동일 패밀리, 다국어 cross-encoder, 경량 (§17-22) |
| **LLM** | `Qwen/Qwen2.5-7B-Instruct-AWQ` | 4-bit AWQ 양자화, T4 15GB 적재 가능. 원본과 동일 아키텍처 (§38-42) |
| **LLM 서빙** | vLLM (Google Colab) | Colab T4 GPU에서 vLLM 서빙 + ngrok 터널. PagedAttention, OpenAI API 호환 (§30-37) |
| **배포 환경** | Google Colab 무료 T4 + ngrok | 세션 최대 12시간, idle 90분. 비용 $0 |
| **검증** | Pydantic v2 | `model_validate_json()`, 상세 에러 → 재시도 피드백, Partial JSON (§55-59) |
| **검색** | Hybrid Dense(0.7) + Sparse(0.3) | 의미 유사도 + 키워드 정확 매칭 결합 (§25) |
| **청킹** | 구조적 분할 + 토큰 기반 (max 1000) | 한국어 토큰 특성, 제목 보존, 재귀적 분할 (§50) |
| **가드레일** | 입력/출력 2계층 | 주제 범위 체크, 할루시네이션 체크, 위험 운동 필터 (§52) |

---

## 3. 디렉토리 구조

```
personal-fit-ai/
│
├── recommend_data/                        # 추천 시스템 학습/참조 데이터 (원본 보존)
│   ├── GYM.csv                            # 목표/BMI별 운동+식단 매핑 (8만 행)
│   ├── exercise.csv                       # 운동 카탈로그 (201개)
│   ├── free_exercise_db.json              # 873개 운동 상세 DB (수행 가이드 포함, Public Domain)
│   ├── Physical activity.html             # WHO 신체 활동 가이드라인
│   └── mturkfitbit_export_.../            # Fitbit 사용자 활동 데이터
│       └── Fitabase Data .../
│           ├── dailyActivity_merged.csv   #   일일 활동 (걸음, 거리, 칼로리)
│           ├── heartrate_seconds_merged.csv #  심박수 (초 단위, 115만 행)
│           ├── weightLogInfo_merged.csv   #   체중/BMI 기록
│           ├── minuteSleep_merged.csv     #   수면 데이터
│           ├── hourlyCalories_merged.csv  #   시간별 칼로리
│           ├── hourlySteps_merged.csv     #   시간별 걸음 수
│           ├── hourlyIntensities_merged.csv # 시간별 운동 강도
│           ├── minuteStepsNarrow_merged.csv # 분별 걸음 수
│           ├── minuteCaloriesNarrow_merged.csv # 분별 칼로리
│           ├── minuteIntensitiesNarrow_merged.csv # 분별 운동 강도
│           └── minuteMETsNarrow_merged.csv #  분별 MET 값
│
├── data/
│   ├── raw/                              # 원본 데이터 (읽기 전용)
│   │   ├── catalog/
│   │   │   └── exercise.csv              # 운동 카탈로그 (201개)
│   │   ├── knowledge/                    # RAG 지식 베이스 원본 문서
│   │   │   ├── exercises/                # 운동별 상세 가이드 (md/PDF)
│   │   │   ├── programs/                 # 트레이닝 프로그램 자료
│   │   │   └── safety/                   # 안전/부상 예방 자료
│   │   └── user/                         # 사용자 관련 데이터
│   │       ├── user_inputs.csv           # 사용자 프로필
│   │       └── workout_events.csv        # 운동 이력 로그
│   └── docs/                             # 가공된 문서 (document_loader.py 출력)
│       ├── exercise_docs.json
│       ├── exercise_detail_docs.json     # free_exercise_db.json → 873개 운동 수행 가이드
│       ├── gym_routine_docs.json         # GYM.csv에서 변환된 루틴 문서
│       ├── guideline_docs.json           # WHO 가이드라인에서 변환된 문서
│       ├── program_docs.json
│       ├── safety_docs.json
│       └── substitution_docs.json
│
├── vectorstore/                          # FAISS 인덱스 (build_vector_index.py 출력)
│   ├── dense.index                       # FAISS IndexFlatIP(1024)
│   ├── sparse_weights.pkl                # BGE-M3 lexical weights
│   └── doc_meta.json                     # 청크별 메타데이터 (제목, 출처, 날짜)
│
├── src/
│   ├── rag/
│   │   ├── chunker.py                    # 문서 청킹 (§50)
│   │   ├── document_loader.py            # 원본 → JSON 변환 (CSV/HTML/md/PDF 지원)
│   │   ├── embedder.py                   # BGE-M3 임베딩 (§23-29)
│   │   ├── vector_store.py               # FAISS 인덱스 관리 (§10-16)
│   │   ├── retriever.py                  # Hybrid 검색 (§25)
│   │   ├── reranker.py                   # Cross-encoder 리랭킹 (§17-22)
│   │   └── prompt_builder.py             # 프롬프트 구성 + token budget (§51)
│   ├── llm/
│   │   └── vllm_client.py                # vLLM OpenAI API 클라이언트 (§30-37)
│   ├── validation/
│   │   ├── routine_schema.py             # Pydantic 모델 정의 (§55-59)
│   │   └── routine_validator.py          # 검증 파이프라인 + 재시도 (§58)
│   ├── guardrails/
│   │   ├── input_guard.py                # 입력 가드레일 (§52)
│   │   └── output_guard.py               # 출력 가드레일 (§52)
│   ├── feedback/                          # [2단계]
│   │   └── collector.py                  # Implicit feedback 수집 (§48)
│   └── app/
│       └── recommend_service.py          # 파이프라인 오케스트레이션
│
├── scripts/
│   └── build_vector_index.py             # 벡터 인덱스 빌드 스크립트
│
├── eval/                                  # [2단계~]
│   ├── test_cases.json                   # 평가 세트 (Q&A 쌍)
│   ├── eval_retrieval.py                 # 검색 평가 (Recall@K, MRR)
│   └── eval_generation.py                # 생성 평가 (Faithfulness, Schema)
│
├── notebooks/
│   └── colab_llm_server.ipynb            # Colab vLLM 서버 실행 노트북
│
├── config.py                             # 전역 설정
├── main.py                               # 엔트리포인트
├── requirements.txt
├── prototype_fit_ai.py                   # 기존 프로토타입 (참고용 보존)
├── RESEARCH.md
├── PLAN.md
├── ERROR.md                  # 런타임 에러 원인 및 해결방법 기록
└── CLAUDE.md
```

### 3.1 기존 PLAN.md 대비 변경점

| 변경 | 이유 |
|------|------|
| `recommend_data/` 추가 | 추천 시스템 학습/참조 데이터 (GYM.csv, WHO 가이드라인, Fitbit 데이터) — `document_loader.py`가 읽어서 RAG 문서로 변환 |
| `data/raw/` 하위에 `catalog/`, `knowledge/`, `user/` 세분화 | 역할별 분리: 카탈로그 vs 지식 문서 vs 사용자 데이터 |
| `data/docs/`에 `gym_routine_docs.json`, `guideline_docs.json` 추가 | `recommend_data/`에서 변환된 RAG 문서 저장 |
| `src/rag/chunker.py` 추가 | Cookbook §50 — 별도 청킹 모듈 필요 |
| `src/guardrails/` 추가 | Cookbook §52 — 입력/출력 안전장치 |
| `src/feedback/` 추가 | YouTube §48 — Implicit feedback 수집 |
| `notebooks/colab_llm_server.ipynb` 추가 | Google Colab T4 GPU에서 vLLM 서버 실행 + ngrok 터널 노트북 |
| `eval/` 추가 | Cookbook §53 — 평가 기반 개발 |
| `config.py` 추가 | 하드코딩 제거, 모든 설정 중앙화 |
| `vectorstore/` 파일 명세 구체화 | dense.index + sparse_weights.pkl + doc_meta.json |

---

## 4. 전체 파이프라인 흐름도

```
어플의 첫 화면에서 사용자 정보 입력
  [가입 시 1회] 나이, 성별, BMI, 골격근량(선택), 운동경력, 목표, 부상이력
  [매 접속 시] 가용시간, 장비, 어제 실시한 운동
→ 아래의 과정을 거쳐서 운동 루틴 추천
    │
    ▼
┌── 입력 가드레일 (input_guard.py) ──────────────┐
│  주제 범위 체크: 피트니스 관련 질문인가?          │
│  부적절 입력 필터: 프롬프트 인젝션 탐지           │
└────────────────────────────────────────────────┘
    │ (통과)
    ▼
┌── BGE-M3 encode (embedder.py) ─────────────────┐
│  쿼리 → dense_vec (1024d) + lexical_weights     │
└────────────────────────────────────────────────┘
    │
    ├──── Dense search (FAISS, Top-20) ────┐
    │                                       │
    ├──── Sparse matching (lexical, Top-20) │
    │                                       │
    ▼                                       ▼
┌── Score 결합 (retriever.py) ───────────────────┐
│  final = 0.7 × dense_score + 0.3 × sparse_score│
│  → Top-10 후보 문서                              │
└────────────────────────────────────────────────┘
    │
    ▼
┌── Re-ranking (reranker.py) ────────────────────┐
│  bge-reranker-v2-m3 cross-encoder               │
│  Top-10 → Top-K (token budget 내)               │
└────────────────────────────────────────────────┘
    │
    ▼
┌── 프롬프트 구성 (prompt_builder.py) ───────────┐
│  시스템 프롬프트 (피트니스 코치 역할)             │
│  + 사용자 프로필 (경력, 목표, 장비, 부상)        │
│  + 검색 문서 (token budget 내 동적 삽입)         │
│  + JSON 출력 스키마                              │
│  + 사용자 질의                                   │
└────────────────────────────────────────────────┘
    │
    ▼
┌── LLM 생성 (vllm_client.py) ──────────────────┐
│  vLLM (Qwen2.5-7B-Instruct-AWQ) via Colab+ngrok│
│  temperature=0.4, top_p=0.8, max_tokens=768    │
│  Shared prefix: 시스템 프롬프트 캐싱             │
└────────────────────────────────────────────────┘
    │
    ▼
┌── 출력 검증 (routine_validator.py) ────────────┐
│  1. JSON 파싱 (Pydantic model_validate_json)    │
│  2. 타입/범위 제약 검증                          │
│  3. 도메인 검증 (@field_validator, @model_validator) │
│  실패 시 → 에러 피드백 포함하여 최대 3회 재시도   │
└────────────────────────────────────────────────┘
    │
    ▼
┌── 출력 가드레일 (output_guard.py) ─────────────┐
│  할루시네이션 체크: 검색 문서에 없는 운동?       │
│  위험 운동 필터: 초보자에게 부적절한 운동 경고    │
└────────────────────────────────────────────────┘
    │
    ▼
최종 JSON 루틴 응답
```

---

## 5. 3단계 개발 로드맵

### 5.1 1단계: RAG + LLM (핵심 파이프라인)

사용자 질의에 대해 운동 지식 문서를 검색하고, LLM이 구조화된 루틴을 생성하는 **최소 기능 파이프라인**.

**범위**: 사용자 이력 반영 없이, 질의 + 프로필 정보만으로 추천.

| 구현 순서 | 파일 | 역할 |
|-----------|------|------|
| 1 | `config.py` | 전역 설정 (모델 경로, 파라미터) |
| 2 | `src/rag/document_loader.py` | 원본 문서 → JSON 변환 |
| 3 | `src/rag/chunker.py` | JSON 문서 → 청크 리스트 |
| 4 | `src/rag/embedder.py` | BGE-M3 임베딩 래퍼 |
| 5 | `src/rag/vector_store.py` | FAISS 인덱스 CRUD |
| 6 | `scripts/build_vector_index.py` | 인덱스 빌드 스크립트 |
| 7 | `src/rag/retriever.py` | Hybrid 검색 |
| 8 | `src/rag/reranker.py` | Cross-encoder 리랭킹 |
| 9 | `src/rag/prompt_builder.py` | 프롬프트 구성 |
| 10 | `src/llm/vllm_client.py` | vLLM 클라이언트 |
| 11 | `src/validation/routine_schema.py` | Pydantic 모델 |
| 12 | `src/validation/routine_validator.py` | 검증 + 재시도 |
| 13 | `src/guardrails/input_guard.py` | 입력 가드레일 |
| 14 | `src/guardrails/output_guard.py` | 출력 가드레일 |
| 15 | `src/app/recommend_service.py` | 파이프라인 오케스트레이션 |
| 16 | `main.py` | CLI 엔트리포인트 |
| 17 | `requirements.txt` | 의존성 |

### 5.2 2단계: RAG + User Logs (사용자 이력 반영)

사용자의 과거 운동 이력(완료율, 건너뛴 운동, 선호 운동)을 프롬프트에 반영하여 개인화 수준 향상.

**추가/수정 파일**:

| 파일 | 변경 내용 |
|------|----------|
| `src/feedback/collector.py` | Implicit feedback 수집 (완료율, 건너뜀, 선호) — §48 |
| `src/rag/prompt_builder.py` | `user_history` 주입 (최근 루틴, 근육 부위별 빈도, 건너뛴 운동) |
| `eval/` | 평가 세트 구축 + Retrieval/Generation 평가 — §53 |

### 5.3 3단계: RAG + Recommender Models (추천 모델 통합)

YouTube 2단계 아키텍처(§44)를 차용하여, 검색 결과에 학습된 랭킹 모델을 결합.

**추가/수정 파일**:

| 파일 | 변경 내용 |
|------|----------|
| `src/rag/retriever.py` | Example Age 시간 가중치 적용 — §48.4 |
| `src/rag/reranker.py` | LightGBM 랭킹 모델 + cross-encoder 앙상블 |
| `src/feedback/collector.py` | 사용자 균등 가중 로직 — §47 |

---

## 6. 핵심 파일별 구현 방식 (1단계)

### 6.1 `config.py` — 전역 설정

모든 하드코딩 값을 중앙화. 환경변수 또는 이 파일 하나만 수정하면 전체 파이프라인 파라미터 변경 가능.

```python
import os

# --- 모델 ---
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# --- vLLM ---
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")

# --- 검색 ---
RETRIEVAL_TOP_N = 20          # 초기 검색 후보 수
RERANK_TOP_K = 5              # 리랭킹 후 최종 문서 수
W_DENSE = 0.7                 # Hybrid search 가중치
W_SPARSE = 0.3

# --- 청킹 ---
CHUNK_MAX_TOKENS = 1000       # 청크 최대 토큰 수

# --- LLM 생성 ---
MAX_MODEL_LEN = 4096          # vLLM 서버 max_model_len
TEMPERATURE = 0.4             # 루틴 일관성 우선
TOP_P = 0.8
REPETITION_PENALTY = 1.05
MAX_TOKENS = 768              # 응답 최대 토큰 (입력 한도 확보: 4096-768=3328)

# --- 검증 ---
MAX_VALIDATION_RETRIES = 3    # Pydantic 검증 실패 시 재시도 횟수

# --- 경로 ---
RAW_CATALOG_DIR = "data/raw/catalog"
RAW_KNOWLEDGE_DIR = "data/raw/knowledge"
RAW_USER_DIR = "data/raw/user"
RECOMMEND_DATA_DIR = "recommend_data"          # 추천 시스템 학습/참조 데이터
DOCS_DIR = "data/docs"
VECTORSTORE_DIR = "vectorstore"
```

### 6.2 `src/rag/document_loader.py` — 원본 → JSON 변환

원본 파일(md, PDF, CSV, HTML)을 읽어 통일된 JSON 문서 리스트로 변환.

**입력**:
- `data/raw/knowledge/` 하위 파일들 (md, PDF)
- `data/raw/catalog/exercise.csv`
- `recommend_data/` 하위 파일들 (CSV, HTML)

**출력**: `data/docs/*.json`

```python
# 출력 JSON 문서 포맷 (1개 문서)
{
    "doc_id": "exercise_bench_press",
    "title": "벤치프레스 완벽 가이드",
    "source": "data/raw/knowledge/exercises/bench_press.md",
    "category": "exercise_guide",     # exercise_guide | program | safety | gym_routine | guideline
    "updated_at": "2026-03-15",
    "text": "벤치프레스는 가슴 근육을 주로 사용하는 복합 운동으로..."
}
```

**구현 방식**:
- `.md` 파일: 직접 읽기
- `.pdf` 파일: `pymupdf` (PyMuPDF) 또는 `pdfplumber`로 텍스트 추출
- `.html` 파일: `BeautifulSoup`으로 본문 텍스트 추출 (JS/CSS 제거)
- `.csv` 파일: 파일별 변환 전략이 다름 (아래 참조)

**`recommend_data/` 파일별 변환 전략**:

| 파일 | 행 수 | 변환 방식 | 출력 | 활용 단계 |
|------|-------|----------|------|----------|
| `GYM.csv` | 8만 | 목표(Goal) × BMI 조합별로 그룹핑 → 각 그룹을 "해당 프로필에 권장되는 운동/식단" 문서로 변환 | `gym_routine_docs.json` | **1단계** — RAG 핵심 지식 |
| `exercise.csv` | 201 | 각 행 → 자연어 운동 설명 문서로 변환 (한국어 운동명 포함) | `exercise_docs.json` | **1단계** — 운동 카탈로그 |
| `free_exercise_db.json` | 873 | 각 운동 → instructions를 본문으로, muscles/equipment/level을 메타데이터로 하는 문서 변환 | `exercise_detail_docs.json` | **1단계** — 운동 수행 가이드 |
| `Physical activity.html` | 1 | HTML → 텍스트 추출 → WHO 가이드라인 문서 | `guideline_docs.json` | **1단계** — 안전/가이드라인 |
| `mturkfitbit_export/` | 11파일 | 1단계에서는 변환하지 않음. 2단계에서 사용자 활동 패턴 분석 참조용으로 활용 | — | **2단계** — 사용자 패턴 |

**`GYM.csv` 변환 상세**:

```python
# GYM.csv 원본 (8만 행, 5개 컬럼)
# Gender, Goal, BMI Category, Exercise Schedule, Meal Plan

# 변환 전략: Goal × BMI Category 조합별 그룹핑 (중복 제거)
# → 고유 조합 수 = ~16개 (4 goals × 4 BMI categories)
# 각 그룹을 하나의 RAG 문서로 변환:

{
    "doc_id": "gym_muscle_gain_normal_weight",
    "title": "근육 증가 목표 — 정상 체중 권장 루틴",
    "source": "recommend_data/GYM.csv",
    "category": "gym_routine",
    "text": "목표: 근육 증가 (muscle_gain)\nBMI 분류: 정상 체중\n\n"
            "권장 운동: Moderate cardio, Strength training, and 5000 steps walking\n\n"
            "권장 식단: Balanced diet with moderate protein and carbohydrates: "
            "Chicken breast, brown rice, spinach, eggs, apple"
}
```

**RESEARCH.md 참조**: §2 (RAG 문서 인덱싱), §50 (Cookbook chunking의 전처리 단계)

### 6.3 `src/rag/chunker.py` — 문서 청킹

JSON 문서를 검색에 최적화된 청크로 분할. RESEARCH.md §50의 5단계 전략 적용.

**입력**: `document_loader.py`가 생성한 JSON 문서 리스트
**출력**: 청크 리스트 (각 청크에 메타데이터 포함)

```python
# 청크 1개의 구조
{
    "chunk_id": "exercise_bench_press_001",
    "doc_id": "exercise_bench_press",
    "title": "벤치프레스 완벽 가이드 > 올바른 자세",   # 부모 제목 보존
    "text": "벤치프레스 완벽 가이드 > 올바른 자세\n\n바벨을 어깨너비로 잡고...",
    "category": "exercise_guide",
    "updated_at": "2026-03-15",
    "token_count": 487
}
```

**청킹 알고리즘** (§50):

```
원본 문서
  ↓
1단계: 섹션 분할 (##, ### 헤더 기준)
  ↓
2단계: 제목 prepend ("상위제목 > 하위제목\n\n본문...")
  ↓
3단계: 토큰 수 체크 (max 1000 토큰)
  ↓ (초과 시)
4단계: 재귀적 분할 (\n\n → \n → . 순서로 경계 탐색)
  ↓
5단계: 노이즈 제거 (참고문헌, 링크 섹션 제외)
  ↓
최종 청크 리스트
```

**토큰 카운트**: `transformers.AutoTokenizer`로 Qwen2.5 토크나이저 사용 (한국어 토큰화 일관성).

**RESEARCH.md 참조**: §50 (Chunking 전략)

### 6.4 `src/rag/embedder.py` — BGE-M3 임베딩

BGE-M3 모델로 텍스트를 Dense + Sparse 벡터로 인코딩.

**입력**: 텍스트 리스트
**출력**: dense 벡터 배열 (N, 1024) + sparse weights 리스트

```python
from FlagEmbedding import BGEM3FlagModel

class Embedder:
    def __init__(self, model_name: str):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)

    def encode(self, texts: list[str]) -> tuple:
        output = self.model.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        return output['dense_vecs'], output['lexical_weights']
```

- `use_fp16=True`: 속도 향상, 정확도 손실 미미 (§26)
- `return_colbert_vecs=False`: 1단계에서는 ColBERT 미사용 (저장 비용 절약)
- encode 1회 호출로 dense + sparse 동시 생성 (§24)

**RESEARCH.md 참조**: §23-29 (BGE-M3)

### 6.5 `src/rag/vector_store.py` — FAISS 인덱스 관리

Dense 벡터 FAISS 인덱스 + Sparse weights를 함께 관리.

**핵심 메서드**:

```python
class VectorStore:
    def __init__(self, dim=1024):
        self.index = faiss.IndexFlatIP(dim)   # §15: 소규모에서 정확도 최대
        self.sparse_weights: list[dict] = []
        self.doc_meta: list[dict] = []

    def add(self, dense_vecs, sparse_weights, metadata): ...
    def search_dense(self, query_vec, top_n) -> tuple: ...  # (scores, indices)
    def get_sparse_weights(self, indices) -> list[dict]: ...
    def save(self, path): ...   # faiss.write_index + pickle + json
    def load(self, path): ...   # faiss.read_index + pickle + json
```

- `IndexFlatIP(1024)`: 내적 기반 정확 검색 (§15)
- L2 정규화 후 Inner Product = 코사인 유사도 (§24.1)
- 확장 계획: 문서 10만+ 시 `IndexIVFPQ`로 전환 (§14)

**RESEARCH.md 참조**: §10-16 (FAISS), §15 (프로젝트 적용)

### 6.6 `scripts/build_vector_index.py` — 인덱스 빌드

전체 문서를 로드 → 청킹 → 임베딩 → 인덱스 저장하는 일괄 스크립트.

```python
# 실행 흐름
def main():
    # 1. document_loader로 원본 문서 로드
    docs = DocumentLoader(RAW_KNOWLEDGE_DIR, RAW_CATALOG_DIR).load_all()

    # 2. chunker로 청킹
    chunks = Chunker(max_tokens=CHUNK_MAX_TOKENS).chunk_all(docs)

    # 3. embedder로 임베딩
    embedder = Embedder(EMBEDDING_MODEL)
    dense_vecs, sparse_weights = embedder.encode([c['text'] for c in chunks])

    # 4. vector_store에 저장
    store = VectorStore(dim=1024)
    store.add(dense_vecs, sparse_weights, [c['metadata'] for c in chunks])
    store.save(VECTORSTORE_DIR)
```

**실행**: `python scripts/build_vector_index.py`
**재실행 조건**: 지식 문서 추가/수정 시

### 6.7 `src/rag/retriever.py` — Hybrid 검색

Dense + Sparse 점수를 결합하여 후보 문서를 검색.

```python
class HybridRetriever:
    def __init__(self, embedder, vector_store, w_dense=0.7, w_sparse=0.3):
        self.embedder = embedder
        self.store = vector_store
        self.w_dense = w_dense
        self.w_sparse = w_sparse

    def retrieve(self, query: str, top_n: int = 20) -> list[dict]:
        # 1. 쿼리 인코딩
        q_dense, q_sparse = self.embedder.encode([query])

        # 2. Dense search
        dense_scores, dense_ids = self.store.search_dense(q_dense, top_n)

        # 3. Sparse score 계산
        sparse_scores = [
            compute_lexical_matching_score(q_sparse[0], self.store.get_sparse_weights(idx))
            for idx in dense_ids[0]
        ]

        # 4. Score 결합
        final_scores = self.w_dense * dense_scores[0] + self.w_sparse * np.array(sparse_scores)

        # 5. 정렬 후 반환 (메타데이터 포함)
        ranked_indices = np.argsort(-final_scores)[:top_n]
        return [self.store.doc_meta[dense_ids[0][i]] | {"score": final_scores[i]}
                for i in ranked_indices]
```

- Dense search에서 초기 후보를 뽑고, 그 후보들에 대해서만 Sparse score 계산 (효율적)
- 가중치 0.7/0.3은 초기값 — 평가 결과에 따라 조정 (§64 체크리스트)

**RESEARCH.md 참조**: §25 (Hybrid Retrieval), §29.3 (구현 스케치)

### 6.8 `src/rag/reranker.py` — Cross-encoder 리랭킹

Bi-encoder(검색)보다 정확한 Cross-encoder로 후보를 재정렬.

```python
from FlagEmbedding import FlagReranker

class Reranker:
    def __init__(self, model_name: str):
        self.model = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        pairs = [[query, doc['text']] for doc in docs]
        scores = self.model.compute_score(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
```

- SBERT 논문의 핵심 교훈: Bi-encoder(빠름, 대규모 검색) → Cross-encoder(정확, 소규모 리랭킹) 2단계 (§21)
- 입력 20개 → 출력 5개 (고정값이 아니라 token budget에 따라 동적 조정 가능)

**RESEARCH.md 참조**: §17-22 (SBERT/Reranker), §26.2 (BGE Reranker 패밀리)

### 6.9 `src/rag/prompt_builder.py` — 프롬프트 구성

검색 문서 + 사용자 정보 + JSON 스키마를 token budget 내에서 조립.

```python
class PromptBuilder:
    def __init__(self, tokenizer, max_model_len: int, max_tokens: int):
        self.tokenizer = tokenizer
        self.system_prompt_budget = 400     # 시스템 프롬프트 예약
        self.doc_budget = max_model_len - max_tokens - self.system_prompt_budget

    def build(self, query: str, docs: list[dict], user_profile: dict = None) -> list[dict]:
        system = self._build_system_prompt(user_profile)
        context = self._insert_docs_within_budget(docs)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\n질문: {query}"}
        ]

    def _insert_docs_within_budget(self, docs: list[dict]) -> str:
        """Token budget 내에서 관련도 순으로 문서 삽입 (§51)"""
        result = ""
        remaining = self.doc_budget
        for doc in docs:
            doc_text = f"\n\n[참고 문서: {doc['title']}]\n{doc['text']}"
            tokens = len(self.tokenizer.encode(doc_text))
            if tokens > remaining:
                break
            result += doc_text
            remaining -= tokens
        return result
```

**시스템 프롬프트 구조**:

```
당신은 전문 피트니스 코치입니다.
아래 참고 문서만을 기반으로 운동 루틴을 추천합니다.
참고 문서에 관련 정보가 없으면 "정보가 부족합니다"라고 답하세요.

[사용자 프로필]  ← 가입 시 1회 입력
- 나이: {age}
- 성별: {gender}
- BMI: {bmi}
- 골격근량: {skeletal_muscle_mass}  ← 없으면 생략
- 운동경력: {experience}
- 목표: {goal}
- 부상이력: {injuries}

[오늘의 조건]  ← 매 접속 시 입력
- 가용 시간: {time}분
- 장비: {equipment}
- 어제 실시한 운동: {yesterday_workout}

반드시 아래 JSON 형식으로만 응답하세요:
{JSON Schema}
```

- "답변 불가" 패턴 포함 (§60)
- JSON Schema를 프롬프트에 포함하여 출력 형식 안내 (§54 방법1)
- token budget으로 문서 수를 동적 결정 — 고정 Top-K가 아님 (§51)

**RESEARCH.md 참조**: §51 (Token Budget), §60 ("답변 불가"), §54 (Structured Outputs)

### 6.10 `src/llm/vllm_client.py` — vLLM 클라이언트

OpenAI API 호환 인터페이스로 vLLM 서버 호출.

```python
from openai import OpenAI
import httpx

class VLLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "dummy"):
        headers = {}
        if "ngrok" in base_url:
            headers["ngrok-skip-browser-warning"] = "true"  # ngrok 무료 tier API 요청 시 필수
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(120.0, connect=10.0),
            default_headers=headers if headers else None,
        )
        self.model = model

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def generate(self, messages: list[dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.4),
            top_p=kwargs.get("top_p", 0.8),
            max_tokens=kwargs.get("max_tokens", 768),
            extra_body={"repetition_penalty": kwargs.get("repetition_penalty", 1.05)},
        )
        return response.choices[0].message.content
```

**vLLM 서버 실행 명령**:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager
```

- `--quantization awq`: AWQ 4-bit 양자화 명시 (T4 15GB 적재용)
- `--gpu-memory-utilization 0.85`: T4에서 안전 마진 확보 (기존 0.9 → 0.85)
- `--enforce-eager`: CUDA graph 비활성화로 1-3GB VRAM 절약 (단일 사용자 개발 환경)
- OpenAI API 호환 → 로컬 GPU 전환 시 URL만 변경 (§37.5)
- Shared prefix: 동일 시스템 프롬프트가 자동으로 KV Cache에서 공유 (§33)
- temperature 0.4: 루틴 일관성 우선 (§40 권장)

**RESEARCH.md 참조**: §30-37 (vLLM), §38-42 (Qwen2.5)

### 6.10.1 `notebooks/colab_llm_server.ipynb` — Colab vLLM 서버

Google Colab에서 실행하는 노트북. Runtime → T4 GPU 선택 후 순서대로 실행.

**셀 1 — vLLM 설치**:

```python
!pip install vllm
```

**셀 2 — vLLM 서버 시작 (백그라운드) 및 준비 대기**:

```python
import subprocess
import time
import requests

proc = subprocess.Popen([...])
# 127.0.0.1 사용 (localhost→IPv6 해석 시 연결 실패 방지, ERROR.md 참조)
for i in range(60):
    try:
        r = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
        if r.status_code == 200:
            print("✅ vLLM 준비 완료")
            break
    except Exception:
        pass
    time.sleep(5)
```

**셀 3 — ngrok 터널 생성**:

```python
!pip install pyngrok
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # https://ngrok.com 무료 발급
tunnel = ngrok.connect(8000)
print(f"VLLM_BASE_URL={tunnel.public_url}/v1")
```

**셀 4 — 헬스체크**:

```python
import requests, time

time.sleep(30)  # 모델 로딩 대기
r = requests.get(f"{tunnel.public_url}/v1/models")
print(r.json())
```

**구현 시 주의 (ERROR.md 참조)**:
- **셀 2**: vLLM 서버 준비 대기 루프에서 `127.0.0.1` 사용 (Colab에서 `localhost`→IPv6 `::1` 해석 시 연결 실패)
- **셀 3**: ngrok 연결 시 `ngrok.connect("127.0.0.1:8000")` 사용 권장 (IPv6 우회)

**사용법**:

1. Colab 노트북 실행 → 셀 3에서 출력된 ngrok URL 복사
2. 로컬 PC에서: CMD `set VLLM_BASE_URL=...` / PowerShell `$env:VLLM_BASE_URL = "..."`
3. `python main.py` 실행

- Colab 무료: T4 15GB, 세션 최대 12시간, idle 90분 timeout
- ngrok 무료: 세션당 8시간, 분당 40 연결 (개발용 충분)

### 6.11 `src/validation/routine_schema.py` — Pydantic 모델

LLM 출력의 JSON 구조를 정의. 타입, 범위, 도메인 규칙을 선언적으로 표현.

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List
from enum import Enum


class Difficulty(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class Exercise(BaseModel):
    name: str = Field(description="운동 이름")
    sets: int = Field(ge=1, le=10)
    reps: str = Field(description="반복 수 (예: '8-12')")
    rest_seconds: int = Field(ge=0, le=300)
    notes: str = ""

    @field_validator("reps")
    @classmethod
    def validate_reps_format(cls, v):
        """'8-12' 또는 '10' 형식만 허용 (§57)"""
        import re
        if not re.match(r"^\d+(-\d+)?$", v):
            raise ValueError(f"reps는 '8-12' 또는 '10' 형식이어야 합니다: {v}")
        return v


class Routine(BaseModel):
    routine_name: str
    difficulty: Difficulty
    target_muscle: str
    exercises: List[Exercise] = Field(min_length=1, max_length=10)
    total_time_minutes: int = Field(ge=10, le=120)
    warnings: List[str] = []

    @model_validator(mode="after")
    def check_time_consistency(self):
        """운동 수 대비 총 시간이 합리적인지 검증 (§57)"""
        min_time = len(self.exercises) * 3   # 운동당 최소 3분
        if self.total_time_minutes < min_time:
            raise ValueError(
                f"운동 {len(self.exercises)}개에 총 {self.total_time_minutes}분은 비현실적입니다"
            )
        return self
```

- `model_json_schema()` → JSON Schema dict → 프롬프트에 포함 가능 (§56.1)
- `Difficulty` enum으로 난이도 값 제한 (§56.1)
- `@field_validator`로 reps 형식 검증, `@model_validator`로 시간 일관성 검증 (§57)

**RESEARCH.md 참조**: §55-59 (Pydantic)

### 6.12 `src/validation/routine_validator.py` — 검증 파이프라인 + 재시도

LLM 출력 → Pydantic 검증 → 실패 시 에러 피드백 포함하여 재시도.

```python
class RoutineValidator:
    def __init__(self, llm_client, prompt_builder, max_retries: int = 3):
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.max_retries = max_retries

    def validate_and_retry(self, raw_output: str, original_messages: list[dict]) -> Routine:
        for attempt in range(self.max_retries):
            try:
                return Routine.model_validate_json(raw_output)
            except ValidationError as e:
                if attempt == self.max_retries - 1:
                    raise
                error_feedback = self._format_errors(e)
                retry_messages = original_messages + [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": f"JSON 검증 실패:\n{error_feedback}\n위 오류를 수정하여 다시 JSON을 생성하세요."}
                ]
                raw_output = self.llm_client.generate(retry_messages)

    def _format_errors(self, e: ValidationError) -> str:
        """ValidationError.errors() → LLM이 이해할 수 있는 피드백 (§58)"""
        lines = []
        for err in e.errors():
            loc = " → ".join(str(l) for l in err["loc"])
            lines.append(f"- {loc}: {err['msg']}")
        return "\n".join(lines)
```

- `model_validate_json()`: `json.loads()` + 타입/범위 검증을 한 번에 수행 (§56.2)
- 에러 피드백을 LLM에 전달하여 자가 수정 유도 (§58)
- Partial JSON 대응: `max_tokens` 도달 시 `from_json(allow_partial=True)` 검토 (§56.3)

**RESEARCH.md 참조**: §58 (전체 검증 파이프라인)

### 6.13 `src/guardrails/input_guard.py` — 입력 가드레일

피트니스와 무관한 질문, 프롬프트 인젝션 등을 차단.

```python
class InputGuard:
    async def check(self, user_input: str) -> tuple[bool, str]:
        """
        Returns: (allowed: bool, reason: str)
        - 키워드 기반 1차 필터 (빠름)
        - 필요 시 LLM 기반 2차 분류 (정확)
        """
        ...
```

- 비동기 실행: RAG 파이프라인과 병렬로 실행하여 레이턴시 추가 없음 (§52.3)
- 1단계에서는 키워드 기반 간단 구현, 정교한 LLM 기반 분류는 2단계에서 추가

**RESEARCH.md 참조**: §52 (Guardrails)

### 6.14 `src/guardrails/output_guard.py` — 출력 가드레일

LLM 응답의 의미적 안전성 검증.

```python
class OutputGuard:
    def check(self, routine: Routine, retrieved_docs: list[dict]) -> tuple[bool, list[str]]:
        """
        Returns: (safe: bool, warnings: list[str])
        1. 할루시네이션 체크: routine.exercises의 운동명이 검색 문서에 존재하는가
        2. 위험 운동 필터: 초보자(beginner)에게 고난이도 운동 추천 시 경고
        """
        ...
```

- 할루시네이션 체크: 검색 문서에 언급되지 않은 운동을 추천하면 경고 (§52.4)
- 위험 운동 필터: 규칙 기반 — 위험 운동 목록 vs 사용자 난이도 교차 체크

**RESEARCH.md 참조**: §52 (Guardrails), §60 ("답변 불가" 처리)

### 6.15 `src/app/recommend_service.py` — 파이프라인 오케스트레이션

전체 파이프라인을 하나의 `recommend()` 메서드로 조합.

```python
class RecommendService:
    def __init__(self, retriever, reranker, prompt_builder,
                 llm_client, validator, input_guard, output_guard):
        ...

    async def recommend(self, query: str, user_profile: dict = None) -> Routine:
        # 1. 입력 가드레일 (비동기)
        guard_ok, reason = await self.input_guard.check(query)
        if not guard_ok:
            raise ValueError(reason)

        # 2. 검색
        candidates = self.retriever.retrieve(query, top_n=RETRIEVAL_TOP_N)

        # 3. 리랭킹
        top_docs = self.reranker.rerank(query, candidates, top_k=RERANK_TOP_K)

        # 4. 프롬프트 구성
        messages = self.prompt_builder.build(query, top_docs, user_profile)

        # 5. LLM 생성
        raw_output = self.llm_client.generate(messages)

        # 6. 검증 + 재시도
        routine = self.validator.validate_and_retry(raw_output, messages)

        # 7. 출력 가드레일
        safe, warnings = self.output_guard.check(routine, top_docs)
        if warnings:
            routine.warnings.extend(warnings)

        return routine
```

### 6.16 `main.py` — CLI 엔트리포인트

```python
def main():
    # 1. config 로드
    # 2. 컴포넌트 초기화 (embedder, vector_store, retriever, ...)
    # 3. 사용자 입력 (query + profile)
    # 4. recommend_service.recommend() 호출
    # 5. 결과 출력 (JSON → 한국어 포맷)
```

### 6.17 `requirements.txt`

```
FlagEmbedding
faiss-cpu
transformers
openai
pydantic>=2.7
pymupdf
beautifulsoup4
pandas
numpy
```

- `faiss-cpu`: GPU 없는 개발 환경용. GPU 서버에서는 `faiss-gpu`로 교체.
- `FlagEmbedding`: BGE-M3 + BGE-Reranker 통합 패키지.
- `openai`: vLLM OpenAI API 호환 클라이언트.
- `pydantic>=2.7`: Partial JSON Parsing 지원 (§56.3).
- `beautifulsoup4`: `Physical activity.html` 등 HTML 문서에서 본문 텍스트 추출.
- `pandas`: `GYM.csv`, Fitbit CSV 등 대용량 CSV 처리.

---

## 7. 데이터 설계

### 7.1 추천 데이터 소스 전체 맵 (`recommend_data/`)

`recommend_data/`는 추천 시스템이 참조하는 **외부 학습/참조 데이터**를 원본 그대로 보관하는 폴더. `document_loader.py`가 이 폴더를 읽어서 RAG 문서로 변환한다.

| 파일/폴더 | 형식 | 행 수 | 내용 | 파이프라인 역할 | 활용 단계 |
|-----------|------|-------|------|---------------|----------|
| `GYM.csv` | CSV | 8만 | Gender × Goal × BMI Category별 운동 스케줄 + 식단 매핑 | **RAG 핵심 지식** — 사용자 프로필 매칭 시 가장 직접적인 참조 문서 | 1단계 |
| `exercise.csv` | CSV | 201 | 운동 카탈로그 (exercise_id, name_ko, pattern, muscle_group, equipment_need) | **운동 메타데이터** — 한국어 운동명 + 카탈로그 | 1단계 |
| `free_exercise_db.json` | JSON | 873 | 운동별 상세 DB — name, level, equipment, primaryMuscles, secondaryMuscles, **instructions**(단계별 수행 가이드), category | **RAG 핵심 지식** — 운동 수행 방법, 자세, 근육군 상세. `exercise.csv`의 영어판 상위 호환 | 1단계 |
| `Physical activity.html` | HTML | 1페이지 | WHO 신체 활동 가이드라인 (주당 권장 활동량, 연령별 기준 등) | **안전/가이드라인 지식** — 과도한 운동 방지, 일반 건강 권고 | 1단계 |
| `mturkfitbit_export/` | CSV×11 | 115만+ | Fitbit 사용자 실제 활동 데이터 (일일 활동, 심박수, 수면, 칼로리, MET, 체중/BMI) | **사용자 행동 패턴 참조** — 현실적인 활동량/칼로리 기준치, 프롬프트 컨텍스트 | 2단계 |

**`GYM.csv` 변환 흐름**:

```
GYM.csv (8만 행)
    ↓
Goal × BMI Category 그룹핑 → ~16개 고유 조합
    ↓
각 조합 → RAG 문서 1개 (운동 스케줄 + 식단 권장)
    ↓
gym_routine_docs.json
    ↓
chunker.py → embedder.py → vectorstore/
```

**`mturkfitbit_export/` 활용 방식 (2단계)**:

| Fitbit 데이터 | 활용 |
|--------------|------|
| `dailyActivity_merged.csv` | 현실적인 일일 활동량 기준치 (평균 걸음 수, 활동 시간) → 프롬프트 컨텍스트 |
| `heartrate_seconds_merged.csv` | 운동 강도별 심박수 범위 참고 → 안전 가이드라인 강화 |
| `weightLogInfo_merged.csv` | BMI 분포/변화 패턴 → 목표 설정 현실성 검증 |
| `minuteSleep_merged.csv` | 수면-운동 상관관계 참고 → 회복 기반 루틴 조정 |
| 나머지 (hourly/minute 데이터) | 시간대별 활동 패턴 → 운동 시간 추천 |

### 7.2 지식 문서 포맷 규약 (`data/raw/knowledge/`)

| 디렉토리 | 내용 | 권장 포맷 |
|----------|------|----------|
| `exercises/` | 운동별 상세 가이드 (자세, 호흡법, 세트/반복 가이드, 주의사항, 대체 운동) | Markdown (.md) |
| `programs/` | 트레이닝 프로그램 (분할법, 주차별 진행, 디로드 가이드) | Markdown (.md) |
| `safety/` | 안전/부상 예방 (워밍업, 근육별 스트레칭, 통증 대응) | Markdown (.md) |

**Markdown 권장 이유**: 헤더 기반 구조적 분할이 자연스러움, chunker.py가 `##`, `###`를 경계로 분할.

**exercise.csv 활용**: `document_loader.py`가 각 행을 최소 문서로 변환:

```
# {name_ko}

- 운동 패턴: {pattern}
- 주요 근육: {muscle_group}
- 필요 장비: {equipment_need}
```

→ 별도 상세 가이드 문서(`data/raw/knowledge/exercises/`)가 있으면 그것이 우선, 없으면 이 최소 문서가 검색 대상.

### 7.3 가공 문서 스키마 (`data/docs/`)

`document_loader.py` 출력. 각 JSON 파일은 문서 리스트:

```json
[
    {
        "doc_id": "exercise_bench_press",
        "title": "벤치프레스 완벽 가이드",
        "source": "data/raw/knowledge/exercises/bench_press.md",
        "category": "exercise_guide",
        "updated_at": "2026-03-15",
        "text": "벤치프레스는 가슴 근육을..."
    }
]
```

| JSON 파일 | 소스 | category 값 |
|-----------|------|-------------|
| `exercise_docs.json` | `exercise.csv` + `data/raw/knowledge/exercises/` | `exercise_guide` |
| `exercise_detail_docs.json` | `recommend_data/free_exercise_db.json` | `exercise_detail` |
| `gym_routine_docs.json` | `recommend_data/GYM.csv` | `gym_routine` |
| `guideline_docs.json` | `recommend_data/Physical activity.html` | `guideline` |
| `program_docs.json` | `data/raw/knowledge/programs/` | `program` |
| `safety_docs.json` | `data/raw/knowledge/safety/` | `safety` |
| `substitution_docs.json` | `data/raw/knowledge/exercises/` (대체 운동 섹션) | `substitution` |

### 7.4 벡터 인덱스 (`vectorstore/`)

| 파일 | 내용 | 생성 주체 |
|------|------|----------|
| `dense.index` | FAISS IndexFlatIP(1024) 바이너리 | `build_vector_index.py` |
| `sparse_weights.pkl` | 청크별 lexical weights (pickle) | `build_vector_index.py` |
| `doc_meta.json` | 청크별 메타데이터 (chunk_id, doc_id, title, category, updated_at) | `build_vector_index.py` |

---

## 8. 주요 트레이드오프

### 8.1 FAISS 인덱스: IndexFlatIP vs IndexIVFPQ

| 항목 | IndexFlatIP | IndexIVFPQ |
|------|------------|------------|
| 검색 정확도 | 100% (brute-force) | ~95% (근사) |
| 검색 속도 (1만 문서) | <10ms | <1ms |
| 메모리 | 40MB (1만 × 1024 × float32) | ~5MB (PQ 압축) |
| **1단계 선택** | **선택** | 확장 시 전환 |

**결정**: 운동 문서 코퍼스는 수천 개 규모 → IndexFlatIP로 충분. 10만+ 확장 시 IndexIVFPQ 전환.
**근거**: §15

### 8.2 Hybrid Search 가중치

| 설정 | Dense 비중 | Sparse 비중 | 특성 |
|------|-----------|------------|------|
| 기본값 | 0.7 | 0.3 | 의미 검색 우선 |
| 키워드 중시 | 0.5 | 0.5 | 정확한 운동명 매칭 강화 |
| 의미 중시 | 0.9 | 0.1 | "가슴 운동" → "벤치프레스" 연결 강화 |

**결정**: 0.7/0.3으로 시작. eval 세트 구축 후 Retrieval Recall@K로 최적값 탐색.
**근거**: §25, §64 체크리스트

### 8.3 JSON 출력 보장: guided_json vs 프롬프트 + Pydantic

| 방법 | 정확도 | 레이턴시 | 유연성 |
|------|--------|---------|--------|
| 프롬프트에 스키마 명시 | 중간 (~80%) | 없음 | 높음 |
| vLLM `guided_json` (outlines) | 매우 높음 (~99%) | +10~30% | 낮음 (스키마 변경 시 재시작) |
| Pydantic 사후 검증 + 재시도 | 높음 (~95%, 3회 시도) | 실패 시 +추가 호출 | 높음 |

**결정**: 프롬프트 스키마 + Pydantic 사후 검증 + 재시도(최대 3회)를 기본으로 채택. 검증 실패율이 높으면 `guided_json` 도입.
**근거**: §54 ("방법 1 + 방법 3을 기본으로")

### 8.4 Temperature

| 값 | 특성 | 용도 |
|----|------|------|
| 0.1~0.3 | 매우 결정적, 반복적 | 동일 질의에 항상 같은 루틴 |
| **0.4** | **일관성 + 약간의 다양성** | **1단계 기본값** |
| 0.7 | 다양한 루틴 생성 | 사용자가 "다른 루틴 추천해줘" 요청 시 |

**결정**: 기본 0.4. "다른 루틴" 요청 시 0.7로 동적 전환 가능하도록 `config.py`에 분리.
**근거**: §40 (Qwen 권장 0.7, 피트니스 도메인은 0.3-0.5 권장)

### 8.5 ColBERT 사용 여부

| 항목 | 사용 시 | 미사용 시 |
|------|--------|----------|
| 검색 정확도 | 높음 | 중간 (Dense+Sparse로도 충분) |
| 저장 비용 | 높음 (토큰별 벡터) | 낮음 |
| 인덱스 빌드 시간 | 김 | 짧음 |

**결정**: 1단계에서는 미사용. Retrieval Recall이 목표치 미달 시 2단계에서 도입 검토.
**근거**: §24.4 (비교 테이블), §64 체크리스트

---

## 9. 미결 의사결정 체크리스트

> RESEARCH.md §64에서 발췌. 구현 중 실험으로 결정할 항목들.

### Embedding & 검색
- [ ] fp16 vs fp32: 정확도 차이 측정 후 결정
- [ ] max_length 설정: 문서 특성에 따라 512~8192 조정
- [ ] Hybrid search 가중치: 초기 0.7/0.3, eval 후 조정
- [ ] Top-K 값: 논문은 5~10, 도메인에 맞게 실험
- [ ] HyDE 적용 여부: Retrieval recall이 낮을 경우 도입 (§62)

### FAISS
- [ ] nprobe 값 (IVFPQ 전환 시): 초기 10, recall 측정 후 조정
- [ ] PQ 파라미터 (확장 시): b=8~16, nbits=8

### LLM 서빙
- [x] GPU 사양 확정: Google Colab 무료 T4 (15GB VRAM)
- [x] 양자화: AWQ 4-bit (~4GB 모델, T4에 여유)
- [x] enforce-eager: CUDA graph 비활성화 (VRAM 절약, 단일 사용자 환경)
- [ ] max_model_len: 4096 (기본값, 프롬프트 크기 측정 후 조정)
- [ ] Shared prefix 캐싱: vLLM 자동 활성화, 효과 측정
- [ ] ngrok 무료 tier 제한: 세션당 8시간, 분당 40 연결
- [ ] Colab 세션 관리: 12시간 제한, idle 90분 timeout 대응

### Structured Outputs
- [ ] guided_json 도입 시점: Pydantic 검증 실패율 >20%이면 도입
- [ ] Partial JSON: max_tokens 잘림 빈도에 따라 `from_json(allow_partial=True)` 적용

### 가드레일
- [ ] 입력 가드레일 방식: 키워드 vs LLM 분류 (1단계는 키워드)
- [ ] 위험 운동 목록 정의

### 평가 (2단계)
- [ ] 평가 세트 규모: 최소 50개 Q&A 쌍
- [ ] Retrieval 목표: Recall@5 >= 0.8
- [ ] Generation 목표: Schema compliance >= 0.95
- [ ] 사용자 균등 가중 로직 (YouTube §47)
