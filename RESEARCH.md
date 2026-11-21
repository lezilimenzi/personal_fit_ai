# RESEARCH.md — 핵심 논문 깊이 분석

# 논문 1: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks(RAG 구조)
> **저자**: Patrick Lewis et al. (Facebook AI Research, UCL, NYU)
> **출처**: NeurIPS 2020, arXiv:2005.11401v4

---

## 1. 논문이 해결하려는 핵심 문제

대규모 사전학습 언어모델(LLM)은 파라미터 안에 사실적 지식을 저장하지만, 다음과 같은 근본적 한계가 있다:

| 한계 | 설명 |
|------|------|
| **지식 접근/조작의 제한** | 파라미터에 암묵적으로 저장된 지식을 정밀하게 꺼내 쓰기 어렵다 |
| **지식 업데이트 불가** | 모델을 재학습하지 않으면 세계 지식을 갱신할 수 없다 |
| **Hallucination** | 사실과 다른 내용을 그럴듯하게 생성한다 |
| **설명 가능성 부재** | 어떤 근거로 답변을 생성했는지 추적이 불가능하다 |

RAG는 **parametric memory**(모델 파라미터)와 **non-parametric memory**(외부 문서 인덱스)를 결합하여 이 문제들을 동시에 해결한다.

---

## 2. RAG 아키텍처 상세

### 2.1 전체 구조

```
Input (x)
    ↓
Query Encoder (BERT_q) → q(x)
    ↓
MIPS (Maximum Inner Product Search) on Document Index
    ↓
Top-K documents [z1, z2, ..., zK] retrieved
    ↓
Generator (BART) receives (x, z) pairs
    ↓
Marginalize over documents → Output (y)
```

### 2.2 두 가지 핵심 컴포넌트

#### Retriever: DPR (Dense Passage Retriever)
- **Bi-encoder 아키텍처** 사용
- **Document encoder** `BERT_d`: 각 문서 z를 dense vector `d(z)`로 변환
- **Query encoder** `BERT_q`: 입력 x를 query vector `q(x)`로 변환
- 관련도 점수: `p(z|x) ∝ exp(d(z)^T · q(x))` (내적 기반)
- Top-K 문서 검색은 **FAISS**를 사용한 MIPS로 sub-linear 시간에 수행
- Wikipedia를 100-word 청크로 분할 → **총 21M 문서**

#### Generator: BART-large
- Pre-trained seq2seq transformer
- 입력: 원본 쿼리 x + 검색된 문서 z를 concatenate
- 문서 제목과 본문을 하나의 시퀀스로 합친 뒤 generator에 전달

### 2.3 두 가지 모델 변형

#### RAG-Sequence
- **하나의 문서**로 전체 출력 시퀀스를 생성
- 수식: `p(y|x) = Σ_{z∈top-k} p(z|x) · Π_i p(y_i|x, z, y_{1:i-1})`
- 검색된 문서 z를 **단일 잠재 변수**로 취급
- 디코딩: 각 문서별로 beam search → 결과를 marginalize ("Thorough Decoding" / "Fast Decoding")

#### RAG-Token
- **각 토큰마다 다른 문서**를 참조 가능
- 수식: `p(y|x) = Π_i Σ_{z∈top-k} p(z|x) · p(y_i|x, z, y_{1:i-1})`
- 여러 문서의 내용을 **토큰 단위로 혼합**하여 응답 생성
- 디코딩: 표준 autoregressive beam search 사용 가능

**핵심 차이**: RAG-Sequence는 문서 수준에서 marginalize, RAG-Token은 토큰 수준에서 marginalize.

---

## 3. 학습 방식

### 3.1 End-to-End 학습
- 목적 함수: 타겟 시퀀스의 **negative marginal log-likelihood** 최소화
- Optimizer: Adam
- **검색된 문서를 잠재 변수(latent variable)로 취급**하여 top-K 근사로 marginalize

### 3.2 학습 시 고정/학습 대상

| 컴포넌트 | 학습 여부 | 이유 |
|----------|----------|------|
| Query Encoder (BERT_q) | **학습 O** | 태스크에 맞는 검색 질의 생성 |
| Document Encoder (BERT_d) | **고정** | 인덱스 재구축 비용이 너무 큼 |
| Document Index | **고정** | BERT_d에 종속 |
| BART Generator | **학습 O** | 태스크별 생성 능력 fine-tuning |

→ Document encoder를 학습하려면 주기적으로 전체 인덱스를 재구축해야 하므로 (REALM 방식), 저자들은 이를 **불필요**하다고 판단하고 고정함.

---

## 4. 실험 결과 요약

### 4.1 Open-domain QA
- Natural Questions: **44.5 EM** (RAG-Sequence) — DPR(41.5), REALM(40.4) 대비 SOTA
- TriviaQA: **68.0** (TQA-Wiki test set)
- 정답이 검색 문서에 없는 경우에도 **11.8% 정확도** (추출형 모델은 0%)

### 4.2 Abstractive QA (MS-MARCO)
- BART 대비 Bleu-1 +2.6, Rouge-L +2.6
- Gold passage 없이도 SOTA에 근접

### 4.3 Jeopardy Question Generation
- RAG-Token이 Q-BLEU-1에서 BART 대비 우세 (22.2 vs 19.7)
- **Human evaluation**: RAG가 BART보다 factual한 경우 42.7% vs 7.1%
- RAG-Token은 여러 문서의 내용을 토큰 단위로 결합 (Figure 2의 posterior 분석)

### 4.4 Fact Verification (FEVER)
- 3-way: 74.5% (gold evidence 없이)
- 검색된 top-1 문서가 gold article인 비율: **71%**, top-10: **90%**

### 4.5 Generation Diversity
- RAG-Sequence의 distinct trigram 비율이 가장 높음 (MSMARCO: 83.5%, Jeopardy: 53.8%)
- BART 대비 유의미하게 다양한 생성

---

## 5. Ablation 분석에서 얻은 인사이트

| 변형 | NQ EM | 시사점 |
|------|-------|--------|
| RAG-Token-BM25 | 29.7 | BM25(sparse) → DPR(dense)로 교체 시 대폭 향상 |
| RAG-Sequence-Frozen | 41.2 | Retriever를 고정해도 상당한 성능, 하지만 joint training이 더 좋음 |
| RAG-Sequence (full) | 44.0 | Query encoder + Generator 공동 학습이 최적 |

- **Dense retrieval이 sparse(BM25)보다 압도적** — 약 +12 EM
- **Retriever를 함께 학습하면** 추가로 +3 EM 개선
- 검색 문서 수: RAG-Sequence는 문서 수 증가에 단조 증가, RAG-Token은 10개 근처에서 peak

---

## 6. 지식 업데이트 실험

- 2016년 vs 2018년 Wikipedia 인덱스를 교체하여 "Who is the President of X?" 질의
- 매칭 인덱스: **68-70% 정확**, 불일치 인덱스: **4-12% 정확**
- → **non-parametric memory를 교체하는 것만으로 모델의 세계 지식을 갱신할 수 있음**

---
---

# 논문 2: FAISS — Billion-scale Similarity Search with GPUs(Vector DB 설계)

> **논문**: *Billion-scale similarity search with GPUs*
> **저자**: Jeff Johnson, Matthijs Douze, Hervé Jégou (Facebook AI Research)
> **출처**: arXiv:1702.08734, 2017
> **오픈소스**: https://github.com/facebookresearch/faiss

---

## 9. 논문이 해결하려는 핵심 문제

RAG 논문에서 MIPS(Maximum Inner Product Search)를 수행할 때 사용한 것이 바로 **FAISS**다. RAG의 non-parametric memory(21M 문서 인덱스)에서 Top-K 문서를 sub-linear 시간에 검색하는 엔진이 FAISS이므로, RAG 파이프라인의 성능은 곧 FAISS의 성능에 직결된다.

| 문제 | 설명 |
|------|------|
| **차원의 저주** | 고차원(50~1000+d) 벡터에서 정확한 최근접 이웃 검색은 brute-force로 불가능 |
| **메모리 한계** | 10억 개 벡터를 원본으로 저장하면 메모리에 들어가지 않음 |
| **GPU 활용 난제** | k-selection 등 핵심 연산이 GPU 병렬성에 맞지 않는 구조 |
| **속도-정확도 트레이드오프** | 근사 검색에서 허용 가능한 정확도 손실로 수 배 속도 향상 필요 |

---

## 10. FAISS의 핵심 인덱싱 구조: IVFADC

### 10.1 전체 아이디어

벡터 y를 **두 단계 양자화**로 압축하여 저장하고, 검색 시에도 압축 상태에서 거리를 계산한다:

```
y ≈ q(y) = q1(y) + q2(y − q1(y))
             ↑            ↑
        coarse quantizer   fine quantizer (PQ)
        (어떤 클러스터?)    (클러스터 내 잔차 벡터 인코딩)
```

### 10.2 Coarse Quantizer (q1) — Inverted File

- k-means로 학습한 **|C1| ≈ √ℓ** 개의 centroid (ℓ = 전체 벡터 수)
- 각 벡터를 가장 가까운 centroid에 할당 → **Inverted List** 구성
- 검색 시 모든 벡터가 아닌, 쿼리에 가까운 **τ개 리스트만 스캔** (multi-probe)

```
10억 벡터, |C1| = √10^9 ≈ 31,623 centroids
→ 평균 리스트 크기 ≈ 31,623 벡터
→ τ=10이면 약 316K 벡터만 스캔 (전체의 0.03%)
```

### 10.3 Fine Quantizer (q2) — Product Quantization

원본 벡터 대신 **b바이트 코드**만 저장하는 압축 기법:

1. d차원 벡터를 **b개의 sub-vector**로 분할: `y = [y₀, y₁, ..., y_{b-1}]`
2. 각 sub-vector를 독립적으로 양자화 (각 256개 reproduction value → 1바이트)
3. 최종 코드: b바이트 (|C2| = 256^b 가지 reproduction value)

**압축 효과 예시**:

| 원본 벡터 크기 | PQ 코드 크기 (b=8) | 압축률 |
|---|---|---|
| 128d × 4bytes = 512B | 8 bytes | **64배** |
| 768d × 4bytes = 3072B | 64 bytes | **48배** |

### 10.4 Asymmetric Distance Computation (ADC)

거리 계산 시 **쿼리는 원본, DB 벡터는 양자화된 코드**를 사용 (비대칭):

```
‖x − q(y)‖² = ‖x̃₁ − q₁(ỹ₁)‖² + ... + ‖x̃_b − q_b(ỹ_b)‖²
```

각 sub-quantizer가 256개 값만 가지므로, **Lookup Table** T₁...T_b (각 256개 엔트리)를 미리 계산:

- 테이블 구축: **256 × d** multiply-adds (1회)
- 거리 계산: **n × b** lookup-adds (벡터당 b번 테이블 참조 + 덧셈)

→ 원본 대비 d/b 배 연산 절약. d=128, b=8이면 **16배 빠름**.

### 10.5 IVFADC 검색 전체 흐름

```
Query x
    ↓
1) Coarse search: τ-argmin ‖x − c‖² (c ∈ C1)
   → τ개의 가장 가까운 inverted list 선택
    ↓
2) Lookup table 계산: T₁...T_b for each selected list
    ↓
3) 선택된 리스트 내 벡터들의 PQ 코드를 순차 스캔
   → 각 코드에 대해 b번 table lookup + 합산 = 근사 거리
    ↓
4) k-selection: 가장 가까운 k개 결과 반환
```

---

## 11. GPU 최적화: WarpSelect k-Selection 알고리즘

### 11.1 왜 k-selection이 병목인가

유사도 검색의 최종 단계는 계산된 거리 배열에서 **가장 작은 k개를 선택**하는 것이다. CPU에서는 max-heap으로 간단하지만, GPU에서는:

- Heap은 **serial tree update** → warp divergence 발생
- 데이터 의존적 메모리 이동 → 불규칙한 메모리 접근 패턴
- 기존 GPU 방법(radix selection, quickselect)은 **다중 패스** 필요

### 11.2 WarpSelect 설계 핵심

| 설계 원칙 | 구현 |
|-----------|------|
| **단일 패스** | 입력 데이터를 한 번만 스캔 |
| **레지스터 기반** | 모든 중간 상태를 GPU 레지스터 파일에 유지 (shared memory 미사용) |
| **Cross-warp 동기화 없음** | 각 warp가 독립적으로 하나의 배열에 대해 k-selection 수행 |
| **커널 퓨전 가능** | 거리 계산 커널에 직접 합칠 수 있음 |

### 11.3 자료구조

```
[Input stream] → 32개씩 읽음 (warp size)
                    ↓
              Thread Queue (각 lane마다 t개 원소, 내림차순)
              → 1차 필터: T⁰_j보다 큰 값은 즉시 버림
                    ↓  (invariant 위반 시)
              Warp Queue (lane-stride register array, k개 원소, 오름차순)
              → 2차 저장: 현재까지 본 가장 작은 k개
```

**불변조건 (invariants)**:
1. 각 lane의 thread queue head(T⁰_j)는 min-k에 포함되지 않음
2. 모든 T⁰_j는 warp queue의 모든 원소보다 큼
3. 지금까지 본 min-k 원소는 thread queue 또는 warp queue에 존재

### 11.4 동작 과정

1. 각 lane이 새 값을 받음 → thread queue head(T⁰_j)와 비교
2. 더 크면 **즉시 폐기** (대부분의 경우)
3. 더 작으면 thread queue에 삽입 정렬
4. **warp ballot**로 어떤 lane이든 불변조건 위반 여부 확인
5. 위반 시 → odd-size sorting/merging network으로 thread queue + warp queue 재정렬

### 11.5 성능 결과

- **이론적 peak 대비 55% 성능** (k=100)
- 기존 GPU SOTA(fgknn) 대비 **1.62배 빠름** (k=100), **2.01배** (k=1000)
- 거리 계산과 **커널 퓨전** 시 기존 대비 25%+ 추가 성능 향상

---

## 12. Multi-GPU 확장 전략

### 12.1 Replication (복제)

- 인덱스 전체를 **R개 GPU에 복제**
- 쿼리를 R등분하여 각 GPU에 분배
- 거의 선형 속도 향상 (nq/R 쿼리씩 처리)

### 12.2 Sharding (분할)

- 인덱스를 **S개 GPU에 분할 저장** (각 GPU에 ℓ/S 벡터)
- 모든 쿼리를 각 shard에 전송 → 부분 결과를 추가 k-selection으로 병합
- 단일 GPU 메모리를 초과하는 대규모 인덱스에 필수

### 12.3 조합

- S개 shard × R개 replica = **S × R GPU** 활용
- 예: Deep1B (10억 벡터) → S=2, R=2 → 4 GPU로 0.0133ms/query 달성

```
[GPU 0: Shard 0, Replica 0]  [GPU 1: Shard 0, Replica 1]
[GPU 2: Shard 1, Replica 0]  [GPU 3: Shard 1, Replica 1]

Query batch → 각 replica에 nq/R 쿼리 분배
           → 각 shard에서 ℓ/S 벡터 대상 검색
           → 부분 결과 병합 → 최종 k-NN
```

---

## 13. 핵심 실험 결과 정리

### 13.1 Billion-scale 검색 성능

| 데이터셋 | 규모 | 메트릭 | FAISS | 기존 GPU SOTA | 속도 비 |
|----------|------|--------|-------|--------------|--------|
| SIFT1M | 100만 | R@1 | 0.80 | 0.51 | 동일 시간 |
| SIFT1B | 10억 | R@10 | 0.376 (17.7μs) | 0.35 (150μs) | **8.5×** |
| Deep1B | 10억 | R@1 | 0.4517 (0.0133ms) | 0.45 (20ms, CPU) | **~1500×** |

### 13.2 k-NN 그래프 구축

| 데이터셋 | 규모 | 소요 시간 | GPU 수 | 비교 |
|----------|------|----------|--------|------|
| Yfcc100M | 9500만 | **35분** | 4 GPU | CPU 128대 클러스터: 108.7시간 |
| Deep1B | 10억 | **12시간 이내** | 4 GPU | 기존 GPU 32대: 200일 추정 |

### 13.3 주요 설계 파라미터와 영향

| 파라미터 | 의미 | 트레이드오프 |
|----------|------|-------------|
| **\|C1\|** (coarse centroids) | inverted list 수 | 많을수록 정밀 → 스캔 범위 축소, 학습 비용 증가 |
| **τ** (multi-probe) | 탐색할 list 수 | 클수록 recall 증가 → 검색 시간 증가 |
| **b** (PQ sub-vectors) | 코드 바이트 수 | 클수록 정확 → 메모리/연산 증가 |
| **k** (결과 수) | 반환할 이웃 수 | k-selection 비용에 직접 영향 |

---

## 14. Product Quantization 심화 — OPQ (Optimized PQ)

논문에서 언급하는 **OPQ(Optimized Product Quantization)**:

- PQ의 sub-vector 분할이 데이터 분포에 최적이 아닐 수 있음
- OPQ는 입력 벡터에 **회전 변환(직교 행렬)**을 적용하여 sub-space 간 상관관계를 최소화
- **전처리 단계**로 적용 가능 — 인덱싱 구조 변경 없이 정확도 향상
- Deep1B 실험에서 d=96 → OPQ로 d=80으로 변환 후 PQ 적용

---

## 15. 우리 프로젝트에의 적용 시사점

### 15.1 FAISS 컴포넌트와 프로젝트 매핑

| FAISS 컴포넌트 | 우리 프로젝트 매핑 | 비고 |
|---|---|---|
| IVFADC 인덱스 | `src/rag/vector_store.py` | 인덱스 생성/검색 API 래핑 |
| PQ 코드 인코딩 | `scripts/build_vector_index.py` | 문서 임베딩 → PQ 코드 변환 |
| Coarse quantizer (k-means) | 인덱스 학습 시 자동 수행 | |C1| 파라미터 설정 필요 |
| Multi-probe (τ) | `src/rag/retriever.py` 검색 파라미터 | τ = nprobe 파라미터 |
| OPQ 전처리 | `src/rag/embedder.py` 후처리 | 선택적 적용 |

### 15.2 우리 규모에서의 현실적 선택

우리 프로젝트의 문서 수는 수천~수만 건 수준으로, 논문의 10억 건과는 완전히 다른 스케일이다:

| 규모 | 권장 인덱스 | 이유 |
|------|------------|------|
| **< 10K 문서** | `IndexFlatL2` 또는 `IndexFlatIP` | 정확한 brute-force가 충분히 빠름 |
| **10K ~ 100K** | `IndexIVFFlat` | coarse quantizer만 사용, PQ 없이 원본 벡터 유지 |
| **100K ~ 1M** | `IndexIVFPQ` | PQ 압축 + inverted file |
| **1M+** | `IndexIVFPQ` + OPQ | 최대 정확도를 위한 전처리 추가 |

→ **1단계(RAG+LLM)에서는 `IndexFlatIP` 또는 `IndexIVFFlat`으로 시작하는 것이 합리적.**
→ 문서가 늘어나면 `IndexIVFPQ`로 마이그레이션.

### 15.3 FAISS 인덱스 파라미터 가이드

```python
import faiss

d = 768          # 임베딩 차원 (사용 모델에 따라)
nlist = 100      # coarse centroid 수 (√n 규칙)
m = 8            # PQ sub-vector 수 (d의 약수)
nbits = 8        # 각 sub-quantizer 비트 수 (보통 8 = 256 centroids)

# 1단계: 소규모 — 정확한 검색
index = faiss.IndexFlatIP(d)

# 확장 시: IVF + PQ
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.nprobe = 10  # τ (multi-probe)
```

### 15.4 논문에서 가져올 핵심 설계 원칙

1. **단계적 인덱스 업그레이드**: 처음부터 복잡한 IVFPQ를 쓸 필요 없음. Flat → IVFFlat → IVFPQ 순서로 필요 시 전환.

2. **nprobe(τ)는 런타임 튜닝 가능**: 인덱스 재구축 없이 검색 시 τ값만 바꿔서 속도-정확도 트레이드오프 조절 가능.

3. **PQ의 b값은 정확도와 메모리의 핵심 레버**: b가 클수록 정확하지만 메모리/연산 증가. 우리 규모에선 b=8~16이면 충분.

4. **인덱스 저장/로드 지원**: `faiss.write_index()` / `faiss.read_index()`로 `vectorstore/` 디렉토리에 영속화.

5. **GPU 가속은 선택적**: CPU 버전으로 시작하고, 필요 시 `faiss.index_cpu_to_gpu()`로 GPU 전환 가능.

---

## 16. 우리 프로젝트(Personal Fit AI)에의 통합 적용 시사점

### 16.1 두 논문의 매핑: RAG 파이프라인 전체 → 프로젝트 구조

| 파이프라인 단계 | RAG 논문 | FAISS 논문 | 우리 프로젝트 |
|---|---|---|---|
| 문서 저장 | Wikipedia 21M chunks | IVFADC 인덱스 (PQ 코드) | `data/docs/` → `vectorstore/` |
| 문서 인코딩 | BERT_d (Document Encoder) | Product Quantization | `src/rag/embedder.py` |
| 쿼리 인코딩 | BERT_q (Query Encoder) | — (쿼리는 원본 사용, ADC) | `src/rag/retriever.py` |
| 유사도 검색 | MIPS (Top-K) | IVFADC + WarpSelect k-selection | `src/rag/vector_store.py` |
| 검색 후처리 | — | — | `src/rag/reranker.py` |
| 응답 생성 | BART Generator (marginalize) | — | `src/llm/vllm_client.py` |
| 프롬프트 구성 | concatenate(x, z) | — | `src/rag/prompt_builder.py` |
| 출력 검증 | — | — | `src/validation/` |

### 16.2 RAG 논문에서 가져올 핵심 설계 원칙

1. **Dense retrieval 필수**: BM25 대비 dense retrieval이 12+ EM 우세. 우리도 반드시 dense embedding 기반 검색 사용.

2. **Query encoder는 학습, Document encoder는 고정**: 문서 인덱스 재구축 비용을 피하면서도 태스크 적응 가능. 우리는 LLM을 직접 fine-tune하지 않더라도, 프롬프트 엔지니어링으로 같은 효과를 낼 수 있음.

3. **RAG-Token 방식이 복합 정보 결합에 유리**: 운동 루틴은 여러 문서(운동 종류, 안전 정책, 대체 운동 등)를 결합해야 하므로, 여러 문서를 참조하는 RAG-Token 스타일이 적합.

4. **Non-parametric memory 교체로 지식 업데이트**: 새로운 운동 연구나 안전 가이드라인이 나오면 문서만 교체하면 됨. 모델 재학습 불필요.

5. **문서를 100-word 청크로 분할**: 우리 도메인 문서도 적절한 크기로 chunking 필요.

### 16.3 FAISS 논문에서 가져올 핵심 설계 원칙

1. **단계적 인덱스 업그레이드**: Flat → IVFFlat → IVFPQ 순서로 규모에 따라 전환.
2. **nprobe(τ)로 런타임 속도-정확도 조절**: 인덱스 재구축 없이 검색 파라미터만 변경.
3. **PQ 코드 크기(b)는 메모리-정확도의 핵심 레버**: 우리 규모에선 b=8~16이면 충분.
4. **인덱스 영속화**: `faiss.write_index()` / `faiss.read_index()`로 `vectorstore/`에 저장.
5. **CPU 우선, GPU 선택적**: `faiss.index_cpu_to_gpu()`로 필요 시 전환.

### 16.4 논문 대비 우리 프로젝트의 차별점

1. **Reranker 추가**: 두 논문 모두 retriever → generator 직결이지만, 우리는 도메인 정확도를 위해 reranking 단계 추가.
2. **Validation layer**: 논문들은 자유 텍스트 생성이 최종 출력이지만, 우리는 구조화된 운동 루틴(sets, reps, weight, rest)을 출력하므로 스키마 검증이 필수.
3. **도메인 특화 문서**: Wikipedia 범용 지식 대신 운동/피트니스 도메인 문서(exercise catalog, safety policy, substitution rules)에 집중.
4. **소규모 인덱스**: 10억 건이 아닌 수천~수만 건 → 복잡한 IVFPQ 대신 Flat 인덱스로 시작 가능.
5. **3단계 점진적 개발**: 단일 시스템이 아닌 RAG+LLM → RAG+user logs → RAG+recommender로 확장.

---
---

# 논문 3: Sentence-BERT — Bi-encoder vs Cross-encoder의 이론적 기반

> **논문**: *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
> **저자**: Nils Reimers, Iryna Gurevych (TU Darmstadt)
> **출처**: arXiv:1908.10084, EMNLP 2019
> **코드**: https://github.com/UKPLab/sentence-transformers

---

## 17. 논문이 해결하려는 핵심 문제

BERT는 문장 쌍 태스크(STS, NLI 등)에서 SOTA 성능을 달성했지만, **구조적으로 대규모 검색에 사용 불가능**했다:

| 문제 | 구체적 수치 |
|------|-----------|
| **O(n²) 계산 복잡도** | 10,000 문장에서 가장 유사한 쌍 찾기: **n(n-1)/2 = ~5천만 번** 추론 |
| **실시간 불가** | 위 연산에 V100 GPU로 **~65시간** 소요 |
| **독립적 문장 임베딩 불가** | BERT는 두 문장을 동시에 입력해야 비교 가능 (cross-encoder) |
| **기존 우회 방법 성능 저하** | BERT 출력의 평균(avg pooling)이나 [CLS] 토큰 → GloVe 평균보다도 **나쁜** 성능 |

---

## 18. 핵심 개념: Cross-encoder vs Bi-encoder

이 논문이 RAG 파이프라인의 **Retriever + Reranker 2단계 구조**를 이해하는 데 핵심적인 이유는, 두 가지 아키텍처의 트레이드오프를 명확히 밝혔기 때문이다:

### 18.1 Cross-encoder (BERT 원본 구조)

```
[Sentence A] [SEP] [Sentence B]
           ↓
     BERT (12/24 layers)
           ↓
   Similarity Score (단일 값)
```

- 두 문장을 **동시에 입력** → 내부 attention으로 단어 수준 직접 비교
- **정확도 최고**: 양방향 attention이 문장 간 세밀한 관계 포착
- **치명적 단점**: 비교할 쌍마다 매번 전체 추론 → O(n²)
- **사용처**: **Reranker** — 소수 후보(Top-N)에 대해서만 정밀 재순위

### 18.2 Bi-encoder (SBERT 구조)

```
Sentence A          Sentence B
    ↓                   ↓
  BERT               BERT        (가중치 공유 — Siamese)
    ↓                   ↓
 Pooling             Pooling
    ↓                   ↓
 벡터 u              벡터 v
    ↓                   ↓
    cosine-sim(u, v)
```

- 각 문장을 **독립적으로** 고정 크기 벡터로 인코딩
- 문장 임베딩을 **미리 계산하여 인덱싱** 가능 (FAISS 등)
- **속도 혁신**: 10,000 문장 비교가 65시간 → **5초**
- **약간의 정확도 손실**: 문장 간 직접 attention 불가
- **사용처**: **Retriever** — 전체 문서 집합에서 빠른 후보 검색

### 18.3 핵심 트레이드오프 정리

| 속성 | Cross-encoder (Reranker) | Bi-encoder (Retriever) |
|------|------------------------|----------------------|
| 입력 | 문장 쌍 동시 | 각 문장 독립 |
| 정확도 | **최고** (단어 수준 attention) | 약간 낮음 |
| 속도 | O(n) per query (Top-N만) | O(1) per query (사전 인덱싱) |
| 확장성 | 전체 DB 대상 불가능 | **10억 건도 가능** (FAISS) |
| 사전 인덱싱 | **불가** | **가능** |
| 클러스터링 | O(n²) — 비실용적 | O(n) — 실용적 |

**결론**: 둘 다 필요하다. **Bi-encoder로 빠르게 후보를 추린 뒤, Cross-encoder로 정밀하게 재순위.**

---

## 19. SBERT 아키텍처 상세

### 19.1 Siamese Network 구조

- **동일한 BERT 가중치**를 두 문장에 각각 적용 (가중치 공유)
- 출력에 **pooling** 적용하여 고정 크기 벡터 생성
- Pooling 전략: **MEAN** (기본, 최고 성능) > CLS > MAX

### 19.2 세 가지 학습 목적 함수

#### 1) Classification Objective (NLI 데이터용)

```
입력: (u, v, |u−v|) → softmax classifier → {entailment, contradiction, neutral}
손실: Cross-entropy
```

- |u−v| (element-wise 차이)가 **가장 중요한 컴포넌트**
- 유사한 쌍은 가깝게, 다른 쌍은 멀게 만드는 효과

#### 2) Regression Objective (STS 데이터용)

```
입력: cosine-sim(u, v) → 실수값
손실: MSE (Mean Squared Error)
```

#### 3) Triplet Objective (유사-비유사 구분)

```
max(‖s_a − s_p‖ − ‖s_a − s_n‖ + ε, 0)

s_a: anchor 문장, s_p: positive, s_n: negative
ε = 1 (margin)
```

### 19.3 학습 세부사항

| 항목 | 값 |
|------|-----|
| 학습 데이터 | SNLI (570K 쌍) + MultiNLI (430K 쌍) |
| Batch size | 16 |
| Optimizer | Adam, lr=2e-5 |
| Warm-up | 학습 데이터의 10% |
| Epochs | 1 |
| **학습 시간** | **~20분** (BERT fine-tuning) |
| Pooling | MEAN (기본) |

---

## 20. 실험 결과에서 얻은 핵심 인사이트

### 20.1 BERT 직접 사용은 끔찍하다

| 방법 | STS 평균 (Spearman ρ) |
|------|---------------------|
| Avg. GloVe embeddings | 61.32 |
| **Avg. BERT embeddings** | **54.81** (GloVe보다 나쁨!) |
| **BERT [CLS] token** | **29.19** (최악) |
| SBERT-NLI-base | **74.89** |
| SBERT-NLI-large | **76.55** |

→ BERT의 출력을 단순 평균/CLS로 사용하면 **GloVe보다도 나쁜** 문장 임베딩이 됨. 반드시 siamese fine-tuning 필요.

### 20.2 SBERT vs Cross-encoder BERT

| 데이터셋 | BERT (cross-encoder) | SBERT (bi-encoder) | 차이 |
|----------|---------------------|-------------------|------|
| STSb (supervised) | 88.77 | 86.15 | -2.62 |
| AFS (10-fold CV) | 78.68 | 77.85 | -0.83 |
| AFS (cross-topic) | 62.02 | 53.82 | **-8.20** |

- 동일 도메인 내에서는 SBERT와 BERT 차이가 **1~3점** 수준
- **도메인 외(cross-topic)에서 BERT가 확실히 우세** → Cross-encoder의 attention이 새로운 패턴에 더 잘 적응
- → **Reranker(cross-encoder)가 도메인 정확도가 중요한 태스크에서 더 가치 있음**

### 20.3 Ablation: 가장 중요한 설계 결정

| 설정 | STS Dev (Spearman) |
|------|-------------------|
| MEAN pooling | **80.78** |
| CLS pooling | 79.80 |
| MAX pooling | 79.07 |
| concatenation (u, v, \|u−v\|) | **80.78** |
| concatenation (u, v) | 66.04 |
| concatenation (\|u−v\|) only | 69.78 |

→ **MEAN pooling**이 최고, **|u−v| (element-wise difference)**가 학습 시 가장 중요한 신호.

### 20.4 계산 효율성

| 방법 | CPU (sent/s) | GPU (sent/s) |
|------|-------------|-------------|
| Avg. GloVe | 6,469 | — |
| InferSent | 137 | 1,876 |
| Universal Sentence Encoder | 67 | 1,318 |
| **SBERT-base** | 44 | 1,378 |
| **SBERT-base (smart batching)** | **83** | **2,042** |

- Smart batching (유사 길이 문장 그룹화): CPU **89%**, GPU **48%** 속도 향상
- GPU에서 SBERT가 InferSent보다 9% 빠르고, Universal Sentence Encoder보다 55% 빠름

---

## 21. Reranker의 이론적 근거: 왜 2단계가 필요한가

이 논문의 핵심 발견을 종합하면, **Retriever + Reranker** 2단계 파이프라인의 이론적 근거가 명확해진다:

### 21.1 단계별 역할

```
전체 문서 DB (수천~수만 건)
    ↓
[1단계: Bi-encoder (BGE-M3 Dense + Sparse)]
    → 각 문서의 임베딩은 사전 계산 (오프라인)
    → 쿼리 임베딩과 FAISS 내적 검색 → ~밀리초
    → Top-20 후보 선정
    ↓
[2단계: Cross-encoder (bge-reranker-v2-m3)]
    → 쿼리 + 각 후보 문서를 쌍으로 BERT에 입력
    → 단어 수준 양방향 attention으로 정밀 비교
    → 20쌍만 처리하므로 속도 문제 없음
    → Top-5 최종 선정
    ↓
LLM에 전달
```

### 21.2 왜 Bi-encoder만으로는 부족한가

논문의 AFS(Argument Facet Similarity) 실험이 이를 직접 증명:

- Cross-topic(새로운 도메인)에서 **BERT(cross-encoder) 62.02 vs SBERT(bi-encoder) 53.82**
- Cross-encoder의 attention은 **문장 간 단어 수준 직접 비교**를 수행
- Bi-encoder는 **각 문장을 독립적으로 인코딩**하므로, 미묘한 의미 차이를 놓칠 수 있음

**우리 프로젝트 관점**: "초보자 벤치프레스 자세" 쿼리에 대해:
- Bi-encoder: "벤치프레스 그립 너비" 문서도 "벤치프레스 중량 선택" 문서도 비슷한 점수
- Cross-encoder: "자세"에 대한 attention이 "그립 너비"(자세 관련) > "중량 선택"(자세 무관)을 정확히 구분

### 21.3 왜 Cross-encoder만으로는 부족한가

| 시나리오 | Cross-encoder 소요시간 |
|----------|---------------------|
| 1,000 문서 × 1 쿼리 | ~13초 (V100 GPU) |
| 10,000 문서 × 1 쿼리 | ~130초 |
| 10,000 문서 × 10,000 문장 유사도 | **~65시간** |

→ 전체 문서 DB에 대해 cross-encoder를 실행하는 것은 **실시간 서비스에서 불가능**.

---

## 22. 우리 프로젝트에의 적용

### 22.1 Retriever + Reranker 매핑

| 역할 | 논문 개념 | 우리 선택 | 프로젝트 파일 |
|------|----------|----------|-------------|
| **Retriever** (1단계) | Bi-encoder (SBERT) | BGE-M3 (Dense + Sparse) | `retriever.py` |
| **Reranker** (2단계) | Cross-encoder (BERT) | bge-reranker-v2-m3 | `reranker.py` |

### 22.2 SBERT 논문에서 가져올 핵심 설계 원칙

1. **BERT 출력을 직접 사용하지 말 것**: avg pooling이나 [CLS]는 GloVe보다 나쁨. 반드시 fine-tuned 모델 사용 (BGE-M3가 이미 이를 해결).

2. **MEAN pooling이 최적**: BGE-M3도 내부적으로 MEAN pooling 기반.

3. **Cross-encoder는 Top-N에만 적용**: 전체 DB 대상 불가. Bi-encoder로 후보를 먼저 추린 뒤 적용.

4. **Smart batching 사용**: 유사 길이 문장 그룹화로 인코딩 속도 최대 89% 향상. 문서 인코딩 시 적용.

5. **도메인 정확도가 중요하면 Reranker 필수**: Cross-topic 실험에서 cross-encoder가 bi-encoder 대비 ~8점 우세. 운동 도메인의 미묘한 구분(자세 vs 중량 vs 호흡 등)에 reranker가 핵심.

---
---

# 모델 분석: BGE-M3 — Multi-Lingual, Multi-Functionality, Multi-Granularity Embedding

> **모델**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
> **논문**: *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*
> **개발**: Beijing Academy of Artificial Intelligence (BAAI)
> **코드**: [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
> **라이선스**: MIT (상업적 사용 완전 무료)

---

## 23. BGE-M3가 해결하는 문제

기존 임베딩 모델들은 각각 **하나의 검색 방식만** 지원하고, **언어/입력 길이에 제한**이 있었다:

| 기존 한계 | BGE-M3의 해결 |
|-----------|--------------|
| Dense retrieval만 지원 (DPR, BGE-v1.5 등) | **3가지 검색 동시 지원**: Dense + Sparse + Multi-vector |
| 영어 또는 단일 언어만 | **100+ 언어** 지원 (한국어 포함) |
| 입력 길이 512 토큰 제한 | **최대 8192 토큰** 처리 가능 |
| 검색 방식별 별도 모델 필요 | **단일 모델**로 통합 |

**M3 = Multi-Functionality + Multi-Linguality + Multi-Granularity**

---

## 24. BGE-M3의 세 가지 검색 기능 상세

### 24.1 Dense Retrieval

- 텍스트를 **단일 1024차원 벡터**로 인코딩
- 쿼리와 문서 간 **내적(dot product)** 또는 **코사인 유사도**로 검색
- RAG 논문의 DPR과 동일한 bi-encoder 패러다임
- FAISS `IndexFlatIP`와 직접 연동 가능

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

sentences = ["벤치프레스 자세 교정", "스쿼트 무릎 통증 예방"]
embeddings = model.encode(sentences)['dense_vecs']
# embeddings.shape: (2, 1024)
```

### 24.2 Sparse Retrieval (Lexical Weight)

- 각 토큰에 대해 **가중치(weight)**를 계산 → BM25와 유사한 역할
- 어휘 기반 매칭으로 **키워드 정확도** 보장
- Dense 임베딩 생성 시 **추가 비용 없이** 동시에 sparse 벡터 생성

```python
output = model.encode(sentences, return_dense=True, return_sparse=True, return_colbert_vecs=False)
lexical_weights = output['lexical_weights']
# {'벤치': 0.23, '프레스': 0.31, '자세': 0.18, '교정': 0.27, ...}
```

**핵심 가치**: Dense가 놓치는 정확한 키워드 매칭을 보완. "벤치프레스"라는 정확한 운동명을 포함한 문서가 반드시 검색됨.

### 24.3 Multi-Vector Retrieval (ColBERT)

- 각 토큰마다 **별도의 벡터**를 생성 (토큰 수 × 차원)
- **Late Interaction**: 쿼리의 각 토큰 벡터와 문서의 각 토큰 벡터 간 MaxSim 계산
- Dense보다 정밀하고, cross-encoder보다 효율적인 **중간 지점**

```python
output_1 = model.encode(["벤치프레스 자세"], return_colbert_vecs=True)
output_2 = model.encode(["벤치프레스 올바른 그립과 자세 가이드"], return_colbert_vecs=True)
score = model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][0])
```

### 24.4 세 가지 검색 방식 비교

| 방식 | 표현 | 정확도 | 속도 | 저장 비용 |
|------|------|--------|------|----------|
| **Dense** | 텍스트 → 1개 벡터 (1024d) | 중간 | 매우 빠름 | 낮음 |
| **Sparse** | 텍스트 → 어휘 가중치 벡터 | 키워드 정확 | 빠름 | 중간 |
| **ColBERT** | 텍스트 → N개 벡터 (토큰별) | 높음 | 중간 | 높음 |

---

## 25. Hybrid Retrieval 파이프라인 (BGE-M3 권장)

BGE-M3 공식 문서에서 RAG 파이프라인으로 **Hybrid Retrieval + Re-ranking**을 권장한다:

```
Query
  ↓
┌─────────────────────────────────────┐
│  BGE-M3 단일 모델 encode 1회       │
│  → dense_vecs (1024d)              │
│  → lexical_weights (sparse)        │
│  → colbert_vecs (multi-vector)     │
└─────────────────────────────────────┘
  ↓                    ↓
Dense search       Sparse search
(FAISS/Milvus)     (역인덱스/BM25 대체)
  ↓                    ↓
└────── Score 결합 ──────┘
  ↓
Top-N 후보 문서
  ↓
Re-ranker (bge-reranker-v2-m3)
  ↓
최종 Top-K 문서 → LLM에 전달
```

### 25.1 가중 결합 스코어

```python
# Dense, Sparse, ColBERT 스코어를 가중 합산
final_score = w[0] * dense_score + w[1] * sparse_score + w[2] * colbert_score
# 기본 가중치: [0.4, 0.2, 0.4]
```

---

## 26. BGE-M3 모델 사양

| 항목 | 값 |
|------|-----|
| **기반 아키텍처** | XLM-RoBERTa (확장) |
| **임베딩 차원** | 1024 |
| **최대 시퀀스 길이** | 8192 토큰 |
| **지원 언어** | 100+ (한국어 포함) |
| **검색 기능** | Dense + Sparse + ColBERT (동시) |
| **라이선스** | MIT (상업 무료) |
| **모델 크기** | ~560M 파라미터 |
| **fp16 지원** | O (속도 향상, 성능 손실 미미) |

### 26.1 BGE 모델 패밀리 비교

| 모델 | 차원 | 최대 길이 | 언어 | 특징 |
|------|------|----------|------|------|
| `bge-m3` | 1024 | 8192 | 다국어 100+ | **3가지 검색 통합**, 최대 성능 |
| `bge-large-en-v1.5` | 1024 | 512 | 영어 | 영어 특화 |
| `bge-base-en-v1.5` | 768 | 512 | 영어 | 중간 크기, 빠름 |
| `bge-small-en-v1.5` | 384 | 512 | 영어 | 경량, 최고 속도 |
| `bge-multilingual-gemma2` | — | — | 다국어 | Gemma-2-9B 기반, 최신 |

### 26.2 BGE Reranker 패밀리

| 모델 | 언어 | 특징 |
|------|------|------|
| `bge-reranker-v2-m3` | 다국어 | 경량 cross-encoder, 빠른 추론 |
| `bge-reranker-v2-gemma` | 다국어 | Gemma 기반, 높은 정확도 |
| `bge-reranker-v2.5-gemma2-lightweight` | 다국어 | 레이어/토큰 압축 지원, 추론 가속 |
| `bge-reranker-large` | 중/영 | 높은 정확도, 느린 추론 |

---

## 27. 학습 방식: Self-Knowledge Distillation

BGE-M3의 핵심 학습 기법:

1. **Self-Knowledge Distillation**: Dense, Sparse, ColBERT 세 가지 출력을 결합한 점수를 **교사 신호(reward signal)**로 사용하여 개별 모드의 성능을 향상
2. **Efficient Batching**: 긴 텍스트 fine-tuning 시 small-batch 전략으로 메모리 효율 극대화
3. **MCLS (Multi-CLS)**: fine-tuning 없이도 긴 텍스트 성능을 향상시키는 기법

→ 하나의 모델이 3가지 검색 모드 모두에서 개별 전문 모델에 필적하는 성능 달성.

---

## 28. 벤치마크 성능

### 28.1 다국어 검색 (MIRACL)
- 100+ 언어에서 **SOTA 수준** 달성
- OpenAI 임베딩 모델 포함 주요 모델들을 능가

### 28.2 교차 언어 검색 (MKQA)
- 한국어 → 영어, 영어 → 한국어 등 교차 언어 검색에서 우수한 성능

### 28.3 긴 문서 검색 (MLDR, NarrativeQA)
- 8192 토큰까지 처리 가능하여 긴 운동 가이드 문서에도 적합

### 28.4 BM25 대비 성능
- Dense retrieval 단독으로도 BM25를 대부분의 언어에서 능가
- **Hybrid (Dense + Sparse)** 사용 시 BM25 대비 확실한 우위

---

## 29. 우리 프로젝트에의 적용

### 29.1 왜 BGE-M3가 최적인가

| 요구사항 | BGE-M3 대응 |
|----------|------------|
| **한국어 지원** | 100+ 언어 중 한국어 포함 ✓ |
| **운동 용어 정확 매칭** | Sparse retrieval로 "벤치프레스", "데드리프트" 등 정확한 키워드 매칭 ✓ |
| **의미적 유사도** | Dense retrieval로 "가슴 운동" ↔ "벤치프레스" 의미 연결 ✓ |
| **긴 문서 처리** | 8192 토큰으로 상세한 운동 가이드 전체 인코딩 가능 ✓ |
| **상업적 사용** | MIT 라이선스 ✓ |
| **단일 모델 운영** | 3가지 검색을 1회 encode로 동시 수행 → 인프라 단순화 ✓ |

### 29.2 프로젝트 매핑

| 프로젝트 컴포넌트 | BGE-M3 활용 |
|---|---|
| `src/rag/embedder.py` | `BGEM3FlagModel` 로딩 및 encode 래핑 |
| `src/rag/vector_store.py` | Dense 벡터 → FAISS `IndexFlatIP(1024)` 저장 |
| `src/rag/retriever.py` | Hybrid search (dense + sparse score 결합) |
| `src/rag/reranker.py` | `bge-reranker-v2-m3`로 re-ranking |
| `scripts/build_vector_index.py` | 전체 문서 batch encode → FAISS 인덱스 + sparse weight 저장 |

### 29.3 구현 스케치: Hybrid Retrieval

```python
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 문서 인코딩 (1회)
docs = ["벤치프레스: 가슴 운동...", "스쿼트: 하체 운동...", ...]
doc_output = model.encode(docs, return_dense=True, return_sparse=True)
dense_vecs = np.array(doc_output['dense_vecs'])    # (N, 1024)
sparse_weights = doc_output['lexical_weights']      # List[Dict]

# FAISS 인덱스 구축
index = faiss.IndexFlatIP(1024)
faiss.normalize_L2(dense_vecs)
index.add(dense_vecs)
faiss.write_index(index, "vectorstore/exercise.index")

# 검색 (Hybrid)
query = "초보자 가슴 운동 추천"
q_output = model.encode([query], return_dense=True, return_sparse=True)
q_dense = np.array(q_output['dense_vecs'])
faiss.normalize_L2(q_dense)

# Dense search
dense_scores, dense_ids = index.search(q_dense, k=20)

# Sparse search (lexical matching)
sparse_scores = [
    model.compute_lexical_matching_score(q_output['lexical_weights'][0], sw)
    for sw in sparse_weights
]

# Score 결합
w_dense, w_sparse = 0.7, 0.3
final_scores = w_dense * dense_scores[0] + w_sparse * np.array([sparse_scores[i] for i in dense_ids[0]])
top_k_indices = dense_ids[0][np.argsort(-final_scores)[:5]]
```

### 29.4 권장 파이프라인 정리

```
사용자 질의 (한국어)
    ↓
BGE-M3 encode (1회)
    ↓
┌──── Dense search (FAISS, Top-20) ────┐
│                                       │
│   Sparse matching (lexical, Top-20)  │
└───────────── Score 결합 ──────────────┘
    ↓
Top-10 후보 문서
    ↓
bge-reranker-v2-m3 (Cross-encoder re-ranking)
    ↓
Top-5 문서
    ↓
LLM (vLLM) → 운동 루틴 생성
    ↓
Validation (스키마 검증)
    ↓
최종 추천
```

---

---
---

# 논문 4: vLLM — Efficient Memory Management for LLM Serving with PagedAttention

> **논문**: *Efficient Memory Management for Large Language Model Serving with PagedAttention*
> **저자**: Woosuk Kwon et al. (UC Berkeley, Stanford, UC San Diego)
> **출처**: SOSP 2023, arXiv:2309.06180
> **오픈소스**: https://github.com/vllm-project/vllm

---

## 30. 논문이 해결하려는 핵심 문제

RAG 파이프라인에서 검색된 문서를 LLM에 전달하여 최종 응답을 생성하는 단계가 있다. 이 LLM 서빙의 효율성이 곧 전체 시스템의 지연 시간과 처리량을 결정한다.

### 30.1 LLM 서빙의 메모리 병목

13B 파라미터 모델을 A100 40GB GPU에서 서빙할 때:

| 메모리 항목 | 비율 | 크기 |
|------------|------|------|
| 모델 파라미터 (고정) | ~65% | ~26 GB |
| **KV Cache (동적)** | **~30%** | ~12 GB |
| Activation (일시적) | ~5% | ~2 GB |

→ **KV Cache 관리가 배치 크기(=처리량)를 결정하는 핵심 병목.**

### 30.2 KV Cache란?

Transformer의 autoregressive 생성 과정에서, 이전 토큰의 **Key/Value 벡터를 캐싱**하여 재계산을 피하는 것:

```
토큰 생성 순서: x1 → x2 → x3 → ... → xn

각 새 토큰 생성 시:
  q_i = W_q · x_i          (새 쿼리만 계산)
  k_1..i, v_1..i            (이전 것은 캐시에서 로드)
  attention: q_i와 k_1..i의 내적 → 가중 평균 v_1..i
```

**KV Cache 크기 예시** (OPT-13B):
- 토큰 1개: 2 × 5120 × 40 × 2bytes = **800 KB**
- 시퀀스 2048 토큰: **1.6 GB**
- → GPU 메모리에 수십 개 요청만 수용 가능

### 30.3 기존 시스템의 문제점

기존 시스템(FasterTransformer, Orca)은 KV Cache를 **연속 메모리**에 저장하며 **최대 길이만큼 사전 할당**:

| 낭비 유형 | 설명 | 비율 |
|-----------|------|------|
| **예약(Reserved)** | 아직 생성되지 않은 미래 토큰 공간을 미리 확보 | ~26.8% |
| **내부 단편화** | 최대 길이(2048) 할당했지만 실제 10 토큰만 생성 | ~13.6% |
| **외부 단편화** | 요청 간 크기 차이로 메모리 빈틈 발생 | ~17.9% |

→ 기존 시스템에서 **실제 유효 메모리 사용률은 20.4%~38.2%** 에 불과.

---

## 31. PagedAttention: OS 가상 메모리에서 영감을 얻은 핵심 알고리즘

### 31.1 OS 가상 메모리와의 대응

| OS 개념 | vLLM 대응 |
|---------|----------|
| 페이지 (Page) | **KV Block** (고정 크기, 기본 16 토큰) |
| 바이트 (Byte) | **토큰** |
| 프로세스 (Process) | **요청 (Request)** |
| 페이지 테이블 | **Block Table** (논리 블록 → 물리 블록 매핑) |
| 가상 주소 공간 | **논리 KV 블록** (연속적) |
| 물리 메모리 | **물리 KV 블록** (비연속 가능) |
| Copy-on-Write | **블록 단위 CoW** (공유 블록 수정 시 복사) |
| 스왑 공간 (Disk) | **CPU RAM으로 스왑** |

### 31.2 핵심 아이디어

```
기존: 요청 A의 KV Cache → [■■■■■□□□□□□□□□□□] (연속 할당, □=낭비)
vLLM: 요청 A의 KV Cache → [Block7][Block1][Block3] (비연속, 필요 시 할당)
```

- KV Cache를 고정 크기 **KV Block**으로 분할
- 블록은 물리 메모리의 **어디에든** 배치 가능 (비연속)
- **Block Table**이 논리 블록 → 물리 블록 매핑을 관리
- 새 토큰 생성 시에만 **동적으로** 물리 블록 할당

### 31.3 PagedAttention 연산

기존 attention을 블록 단위로 변환:

```
기존: o_i = Σ_j a_ij · v_j                (연속 메모리에서 모든 v 접근)

PagedAttention: o_i = Σ_j V_j · A_ij^T    (각 블록 V_j를 개별 페치)
  - K_j, V_j: j번째 블록의 key/value 벡터
  - 블록별로 독립 접근 → 비연속 메모리 OK
```

---

## 32. vLLM 시스템 아키텍처

### 32.1 전체 구조

```
┌────────────────────────────────┐
│     Centralized Scheduler      │
│  (FCFS, preemption, batching)  │
│                                │
│   ┌─── KV Cache Manager ───┐  │
│   │  Block Table 관리        │  │
│   │  GPU Block Allocator    │  │
│   │  CPU Block Allocator    │  │
│   └─────────────────────────┘  │
└───────┬──────┬──────┬──────────┘
        │      │      │  (control messages)
   ┌────▼─┐┌───▼──┐┌──▼───┐
   │Worker││Worker││Worker│
   │  0   ││  1   ││ N-1  │  (GPU workers)
   │Model ││Model ││Model │
   │Shard ││Shard ││Shard │
   │Cache ││Cache ││Cache │
   │Engine││Engine││Engine│
   └──────┘└──────┘└──────┘
```

### 32.2 핵심 컴포넌트

| 컴포넌트 | 역할 |
|----------|------|
| **Centralized Scheduler** | 요청 스케줄링 (FCFS), 선점/재개 결정, 배치 구성 |
| **KV Cache Manager** | Block Table 관리, 물리 블록 할당/해제, CoW 실행 |
| **Block Engine** | GPU DRAM에 물리 블록 풀 관리, CPU RAM 스왑 공간 관리 |
| **Workers** | 모델 샤드 실행, PagedAttention 커널로 KV Cache 접근 |

### 32.3 프론트엔드

- **FastAPI** 기반 HTTP 서버
- **OpenAI API 호환** 인터페이스 (Completion API, Chat API)
- 사용자별 sampling 파라미터 커스터마이징 (temperature, top_p, max_tokens, beam_width 등)

---

## 33. 고급 디코딩 시나리오별 메모리 공유

### 33.1 Parallel Sampling (다중 샘플링)

하나의 프롬프트에서 여러 출력을 생성할 때, **프롬프트의 KV Cache를 공유**:

```
Prompt: "추천 운동 루틴을 알려줘"
  ↓
Sample A: "벤치프레스 3세트..."  → 별도 KV 블록
Sample B: "스쿼트 4세트..."     → 별도 KV 블록
                                  ↑ 프롬프트 블록은 공유 (ref count=2)
```

- 물리 블록에 **reference count** 도입
- 공유 블록 수정 시 **Copy-on-Write**: 새 물리 블록에 복사 후 수정
- 메모리 절약: **6~10%** (parallel sampling), **37~55%** (beam search)

### 33.2 Beam Search

- 여러 빔 후보가 **동적으로 KV 블록을 공유**
- 공유 패턴이 디코딩 진행에 따라 변화 (OS의 fork 트리와 유사)
- 기존 시스템은 빔 간 KV Cache를 매번 복사 → vLLM은 블록 공유로 복사 최소화

### 33.3 Shared Prefix (시스템 프롬프트)

- RAG에서 공통 시스템 프롬프트 사용 시, **prefix KV Cache를 미리 계산하여 공유**
- 여러 요청이 동일 prefix 블록을 참조 → 중복 계산/저장 제거
- 메모리 절약 + 프롬프트 단계 계산 건너뛰기
- **few-shot prefix 공유 시 3.58배 처리량 향상**

---

## 34. 스케줄링과 선점 (Preemption)

### 34.1 문제

- LLM 출력 길이를 사전에 알 수 없음
- 요청이 늘어나면 GPU 물리 블록이 부족해질 수 있음

### 34.2 해결: 두 가지 복구 기법

| 기법 | 방식 | 장점 | 단점 |
|------|------|------|------|
| **Swapping** | KV 블록을 CPU RAM으로 이동 | 재계산 불필요 | PCIe 대역폭에 종속, 작은 블록 시 비효율 |
| **Recomputation** | KV Cache를 처음부터 재계산 | 블록 크기 무관, 일정한 오버헤드 | GPU 계산 필요 |

- 기본 블록 크기(16)에서 두 기법은 **비슷한 end-to-end 성능**
- vLLM은 **FCFS + all-or-nothing eviction** 정책: 가장 나중에 도착한 요청부터 선점

---

## 35. 분산 실행 (Multi-GPU)

- **Megatron-LM 스타일 tensor model parallelism** 지원
- SPMD 실행: 각 GPU worker가 동일 입력을 받되, attention head를 분할 처리
- **단일 KV Cache Manager**가 중앙에서 모든 GPU worker의 블록 테이블 관리
- GPU 간 동기화: **NCCL all-reduce** (스케줄러 개입 없이)

| 모델 크기 | GPU 구성 | 총 GPU 메모리 | KV Cache 용량 |
|-----------|---------|-------------|-------------|
| 13B | 1× A100 | 40 GB | 12 GB |
| 66B | 4× A100 | 160 GB | 21 GB |
| 175B | 8× A100-80GB | 640 GB | 264 GB |

---

## 36. 핵심 실험 결과

### 36.1 처리량 비교

| 비교 대상 | vLLM 처리량 향상 | 조건 |
|-----------|---------------|------|
| **FasterTransformer** 대비 | **최대 22배** | OPT-13B, ShareGPT |
| **Orca (Max)** 대비 | **2.7~8배** | 동일 지연 시간 수준 |
| **Orca (Oracle)** 대비 | **1.7~2.7배** | 동일 지연 시간 수준 |

### 36.2 배치 크기 향상

| 시스템 | 평균 동시 배치 요청 수 (ShareGPT) |
|--------|-------------------------------|
| Orca (Max) | 7.00 |
| Orca (Pow2) | 9.81 |
| Orca (Oracle) | 13.62 |
| **vLLM** | **30.42** (4.3배) |

### 36.3 메모리 효율

| 시스템 | 유효 KV Cache 사용률 |
|--------|-------------------|
| Orca (Max) | **20.4%** |
| Orca (Oracle) | **38.2%** |
| **vLLM** | **96.3%** |

### 36.4 디코딩 알고리즘별 메모리 절약

| 알고리즘 | 설정 | 메모리 절약 |
|----------|------|-----------|
| Parallel sampling | 6개 출력 | 9.8% |
| Beam search | beam width 6 | **55.2%** |
| Shared prefix | 5-shot | 처리량 **3.58배** 향상 |

### 36.5 PagedAttention 커널 오버헤드

- FasterTransformer 대비 attention 커널 지연시간 **20~26% 높음**
- 하지만 attention은 전체 모델 연산의 일부 → end-to-end에서는 메모리 효율 이점이 압도
- 기본 블록 크기 **16**: GPU 병렬성과 내부 단편화의 최적 균형점

---

## 37. 우리 프로젝트에의 적용

### 37.1 왜 vLLM인가

| 요구사항 | vLLM 대응 |
|----------|----------|
| **로컬 LLM 서빙** | OpenAI API 호환 서버로 즉시 사용 가능 |
| **높은 처리량** | PagedAttention으로 동일 GPU에서 2~4배 더 많은 요청 처리 |
| **다양한 모델 지원** | GPT, OPT, LLaMA 등 주요 모델 지원 |
| **분산 실행** | Multi-GPU tensor parallelism으로 대형 모델도 서빙 |
| **Shared prefix** | RAG의 시스템 프롬프트를 prefix로 캐싱하여 중복 계산 제거 |
| **OpenAI API 호환** | `src/llm/vllm_client.py`에서 표준 API로 호출 |

### 37.2 프로젝트 매핑

| vLLM 기능 | 우리 프로젝트 활용 |
|-----------|-----------------|
| OpenAI-compatible API | `src/llm/vllm_client.py` — HTTP 요청으로 LLM 호출 |
| Shared prefix | 시스템 프롬프트 (운동 추천 지침) 캐싱 |
| Sampling parameters | temperature, top_p 조절로 다양성/일관성 제어 |
| Streaming | 실시간 토큰 스트리밍으로 UX 개선 |
| Multi-GPU | 대형 모델 서빙 시 tensor parallelism 활용 |

### 37.3 vLLM 서버 실행 예시

```bash
# vLLM 서버 시작 (OpenAI API 호환)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --port 8000 \
    --max-model-len 4096
```

### 37.4 클라이언트 호출 예시

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "당신은 전문 피트니스 코치입니다. 검색된 문서를 참고하여 운동 루틴을 추천합니다."},
        {"role": "user", "content": f"참고 문서:\n{retrieved_docs}\n\n질문: 초보자 가슴 운동 루틴 추천해줘"}
    ],
    temperature=0.7,
    max_tokens=1024
)
```

### 37.5 vLLM 논문에서 가져올 핵심 설계 원칙

1. **OpenAI API 호환**: vLLM 서버를 띄우고 표준 API로 호출 — 추후 클라우드 API로 전환 시 코드 변경 최소화.

2. **Shared prefix 활용**: 시스템 프롬프트(운동 추천 지침, 안전 가이드)를 prefix로 캐싱하면 동일 프롬프트를 매번 재계산하지 않아 지연 시간 대폭 감소.

3. **Sampling 파라미터 설계**: temperature로 루틴 다양성 제어, max_tokens로 응답 길이 제한.

4. **GPU 메모리 계획**: 모델 크기 + KV Cache 예산을 고려하여 서버 사양 결정. 8B 모델은 단일 GPU, 70B+ 모델은 Multi-GPU 필수.

5. **블록 크기 16 기본값**: vLLM 기본 설정이 대부분의 워크로드에 최적.

---

# 모델 분석: Qwen2.5-7B-Instruct — Open-source LLM 후보

> **모델**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
> **개발**: Alibaba Cloud (Qwen Team)
> **라이선스**: Apache 2.0 (상업적 사용 완전 무료)
> **실행 라이브러리**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), vLLM

---

## 38. Qwen2.5-7B-Instruct 모델 사양

### 38.1 핵심 스펙

| 항목 | 값 |
|------|-----|
| **파라미터 수** | 7.61B (non-embedding: 6.53B) |
| **아키텍처** | Transformer (Causal LM) |
| **컨텍스트 길이** | **128K 토큰** (YaRN으로 32K 이상 확장) |
| **최대 생성 길이** | 8K 토큰 |
| **Attention** | GQA (28 query heads, 4 KV heads) |
| **레이어 수** | 28 Transformer layers |
| **활성화 함수** | SwiGLU |
| **정규화** | RMSNorm |
| **위치 인코딩** | RoPE + YaRN (length extrapolation) |
| **학습 데이터** | **18조(trillion) 토큰** |
| **지원 언어** | **29+ 언어** (한국어 포함) |
| **라이선스** | Apache 2.0 |

### 38.2 벤치마크 성능

| 벤치마크 | 점수 | 설명 |
|----------|------|------|
| **MMLU** (5-shot) | **81.2%** | 일반 지식/추론 |
| **HumanEval** (pass@1) | **68.4~83%** | 코드 생성 |
| **수학 추론** | **80.5%** | 수학 문제 풀이 |

→ 7B 급 모델 중 **최상위권 성능**. 특히 코딩과 수학에서 이전 버전(Qwen2) 대비 대폭 향상.

### 38.3 GQA (Grouped Query Attention) 구조

Qwen2.5는 메모리 효율을 위해 **Grouped Query Attention**을 사용:

```
기존 MHA: Query 28 heads, Key 28 heads, Value 28 heads
GQA:       Query 28 heads, Key  4 heads, Value  4 heads
                                 ↑
                    7개 query head가 1개 KV head 공유
```

- KV Cache 크기가 **1/7**로 감소 → vLLM의 PagedAttention과 시너지
- 더 많은 요청을 동시에 배치 가능

### 38.4 YaRN (Yet another RoPE extensioN)

- 기본 학습 시 32K 토큰 컨텍스트
- YaRN으로 **128K까지 확장** (factor=4.0)
- RAG에서 검색된 여러 문서 + 시스템 프롬프트를 한꺼번에 입력 가능

---

## 39. Hugging Face Transformers — LLM 실행 라이브러리

### 39.1 Transformers란

[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)는 SOTA 머신러닝 모델의 **모델 정의 프레임워크**로, 추론과 학습 모두를 지원한다. 1M+ 모델 체크포인트가 Hugging Face Hub에 공개되어 있다.

**핵심 역할**: 모델 정의를 중앙화하여 다양한 프레임워크(vLLM, SGLang, TGI, DeepSpeed, llama.cpp 등)에서 호환 가능하게 한다.

### 39.2 주요 컴포넌트

| 컴포넌트 | 역할 | 우리 프로젝트 활용 |
|----------|------|-----------------|
| **AutoModelForCausalLM** | LLM 모델 자동 로딩 | Qwen2.5 모델 로딩 |
| **AutoTokenizer** | 토크나이저 자동 로딩 | 입력 텍스트 토큰화 |
| **Pipeline** | 간편 추론 API | 빠른 프로토타이핑 |
| **Chat Template** | 대화 형식 자동 적용 | system/user/assistant 포맷 |
| **generate()** | 텍스트 생성 API | 디코딩 전략 제어 |

### 39.3 두 가지 실행 경로

#### 경로 1: Transformers 직접 실행 (개발/테스트용)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [
    {"role": "system", "content": "당신은 전문 피트니스 코치입니다."},
    {"role": "user", "content": "초보자 가슴 운동 루틴을 추천해줘"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
```

- **장점**: 설치 간단, 디버깅 용이, 모델 수정 가능
- **단점**: 배치 처리/동시 요청 처리에 비효율, KV Cache 관리 없음

#### 경로 2: vLLM 서빙 (프로덕션용)

```bash
# vLLM 서버 시작
vllm serve Qwen/Qwen2.5-7B-Instruct
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "당신은 전문 피트니스 코치입니다."},
        {"role": "user", "content": f"참고 문서:\n{docs}\n\n초보자 가슴 운동 루틴 추천해줘"}
    ],
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=512
)
```

- **장점**: PagedAttention, 동시 요청 처리, OpenAI API 호환, 높은 처리량
- **단점**: 별도 서버 프로세스 필요

### 39.4 권장: 개발 단계별 실행 경로

| 단계 | 실행 방식 | 이유 |
|------|----------|------|
| **프로토타이핑** | Transformers 직접 | 빠른 실험, 디버깅 용이 |
| **통합 테스트** | vLLM 서버 | API 호환성 확인, 프롬프트 튜닝 |
| **프로덕션** | vLLM 서버 | 처리량, 메모리 효율, 안정성 |

---

## 40. Qwen2.5-7B-Instruct의 기본 Sampling 파라미터

Qwen 공식 문서에서 권장하는 기본 디코딩 하이퍼파라미터:

| 파라미터 | 기본값 | 의미 |
|----------|--------|------|
| `temperature` | 0.7 | 생성 다양성 (낮을수록 결정적) |
| `top_p` | 0.8 | Nucleus sampling 확률 임계값 |
| `repetition_penalty` | 1.05 | 반복 억제 (1.0=없음, 높을수록 억제) |
| `max_tokens` | 512 | 최대 생성 토큰 수 |

→ 우리 프로젝트에서는 운동 루틴의 **일관성**이 중요하므로 temperature를 **0.3~0.5**로 낮추는 것을 권장.

---

## 41. 우리 프로젝트에 Qwen2.5-7B-Instruct가 적합한 이유

| 요구사항 | Qwen2.5-7B-Instruct 대응 |
|----------|-------------------------|
| **한국어 지원** | 29+ 언어 중 한국어 포함 ✓ |
| **긴 컨텍스트** | 128K 토큰 — RAG 검색 문서 다수를 한번에 입력 가능 ✓ |
| **구조화된 출력** | JSON 출력 생성 능력 우수 — 운동 루틴 스키마 생성에 적합 ✓ |
| **단일 GPU 서빙** | 7B 파라미터 → A100 40GB 단일 GPU에서 서빙 가능 ✓ |
| **GQA로 메모리 효율** | KV heads 4개 → KV Cache 크기 1/7 감소, vLLM과 시너지 ✓ |
| **상업적 사용** | Apache 2.0 ✓ |
| **vLLM 공식 지원** | Qwen2.5 전용 vLLM 배포 가이드 제공 ✓ |
| **Instruction following** | 18T 토큰 학습 + instruction tuning으로 지시 따르기 우수 ✓ |

### 41.1 다른 후보 모델과의 비교

| 모델 | 크기 | 한국어 | 컨텍스트 | 라이선스 | GQA | 비고 |
|------|------|--------|---------|---------|-----|------|
| **Qwen2.5-7B-Instruct** | 7.6B | ✓ | 128K | Apache 2.0 | ✓ | **최우선 후보** |
| Llama-3-8B-Instruct | 8B | △ (약함) | 8K | Llama 3 License | ✓ | 한국어 약함 |
| Mistral-7B-Instruct | 7.2B | △ | 32K | Apache 2.0 | ✓ | 한국어 미지원 |
| SOLAR-10.7B | 10.7B | ✓ | 4K | Apache 2.0 | ✗ | 컨텍스트 짧음 |

→ 한국어 + 긴 컨텍스트 + 라이선스 + GQA 효율을 모두 만족하는 모델은 **Qwen2.5-7B-Instruct**가 현재 최적.

---

## 42. 프로젝트 RAG 파이프라인에서의 LLM 호출 흐름

```
사용자 질의: "초보자 가슴 운동 추천해줘"
    ↓
BGE-M3 encode → FAISS Dense+Sparse 검색 → Top-20
    ↓
bge-reranker-v2-m3 → Top-5 문서
    ↓
┌─────────────── Prompt Builder ───────────────────┐
│ [System] 당신은 전문 피트니스 코치입니다.           │
│          아래 참고 문서를 기반으로 운동 루틴을      │
│          JSON 형식으로 추천하세요.                  │
│                                                   │
│ [참고 문서 1] 벤치프레스: 가슴 운동, 3~4세트...    │
│ [참고 문서 2] 인클라인 덤벨 프레스: 윗가슴...      │
│ [참고 문서 3] ...                                  │
│                                                   │
│ [User] 초보자 가슴 운동 추천해줘                   │
└──────────────────────────────────────────────────┘
    ↓
vLLM (Qwen2.5-7B-Instruct)
    → PagedAttention으로 효율적 KV Cache 관리
    → Shared prefix: 시스템 프롬프트 캐싱
    ↓
JSON 응답 생성
    ↓
Validation (Pydantic 스키마 검증)
    ↓
최종 운동 루틴 추천
```

---
---

# 논문 분석: Deep Neural Networks for YouTube Recommendations — 추천 시스템 아키텍처

> **저자**: Paul Covington, Jay Adams, Emre Sargin (Google)
> **출처**: RecSys '16, ACM Conference on Recommender Systems (2016)
> **인용 수**: 2,384+ (추천 시스템 분야에서 가장 영향력 있는 논문 중 하나)
> **핵심 의의**: **실제 프로덕션 규모(10억+ 사용자)**에서 작동하는 추천 시스템의 구현을 공개한 논문

---

## 43. 논문이 해결하려는 핵심 문제

YouTube는 **수십억 개의 동영상**에서 **10억+ 사용자** 각각에게 개인화된 추천을 제공해야 한다. 이 문제의 3대 도전 과제:

| 도전 과제 | 설명 | 우리 프로젝트 대응 |
|-----------|------|-----------------|
| **Scale** | 수백만 후보 중 수십 개를 실시간 선별 | 수백~수천 운동 중 5~10개 추천 |
| **Freshness** | 새로운 콘텐츠를 즉시 추천에 반영 | 새로운 운동/루틴 트렌드 반영 |
| **Noise** | 사용자 만족도의 직접 측정이 어려움 (implicit feedback) | 사용자가 루틴을 "완료"했는지, 선호했는지 간접 추론 |

---

## 44. 2단계 아키텍처 — Candidate Generation + Ranking

### 44.1 전체 시스템 구조 (핵심)

```
전체 동영상 코퍼스 (수백만)
    ↓
┌─── 1단계: Candidate Generation ───────────────┐
│  "수백만 → 수백"으로 축소                       │
│  • 사용자 임베딩 기반 Approximate Nearest Neighbor│
│  • Collaborative Filtering (거친 개인화)        │
│  • 빠른 속도 우선 (broad recall)               │
└────────────────────────────────────────────────┘
    ↓ 수백 개 후보
┌─── 2단계: Ranking ────────────────────────────┐
│  "수백 → 수십"으로 정밀 순위 매기기             │
│  • 풍부한 피처 사용 (수백 개)                   │
│  • 세밀한 개인화 (fine-grained)                │
│  • 최종 목표: Expected Watch Time 최대화       │
└────────────────────────────────────────────────┘
    ↓ 수십 개 (최종 추천)
사용자에게 표시
```

### 44.2 왜 2단계인가?

- **1단계만으로는 부족**: 수백만 후보를 모두 풍부한 피처로 스코어링하면 레이턴시 초과
- **2단계만으로는 부족**: 처음부터 정밀 모델로 수백만을 평가할 수 없음
- **2단계 접근의 이점**: 1단계에서 빠르게 후보를 좁히고, 2단계에서 정밀하게 순위 매김

### 44.3 우리 프로젝트에의 매핑

```
운동 문서 코퍼스 (수백~수천)
    ↓
┌─── 1단계: RAG Retrieval (= Candidate Generation) ──┐
│  BGE-M3 임베딩 + FAISS 검색 → Top-20 후보           │
│  + bge-reranker-v2-m3로 → Top-5                    │
│  = YouTube의 "Candidate Generation + Ranking"에 해당│
└─────────────────────────────────────────────────────┘
    ↓ Top-5 운동 문서
┌─── 2단계: LLM Generation (= Final Ranking) ────────┐
│  Qwen2.5-7B-Instruct가 사용자 프로필 + 검색 문서    │
│  기반으로 개인화된 루틴 생성                         │
│  = YouTube의 Ranking에 해당 (최종 점수/순위 결정)    │
└─────────────────────────────────────────────────────┘
    ↓
개인화된 운동 루틴 JSON
```

---

## 45. Candidate Generation — 추천을 분류 문제로 재정의

### 45.1 Extreme Multi-class Classification

YouTube는 추천을 **"사용자가 시간 t에 동영상 i를 시청할 확률"**을 예측하는 분류 문제로 정의한다:

```
P(wt = i | U, C) = exp(vi · u) / Σj∈V exp(vj · u)
```

- `u ∈ R^N`: 사용자+컨텍스트의 임베딩 벡터
- `vi ∈ R^N`: 각 동영상의 임베딩 벡터
- `V`: 전체 동영상 코퍼스 (수백만 클래스)

**학습 시**: Sampled Softmax (수천 개 네거티브 샘플) → 100배 이상 학습 속도 향상
**서빙 시**: Softmax 확률 불필요 → **dot product 기반 Approximate Nearest Neighbor** 검색

### 45.2 모델 아키텍처

```
입력 피처들:
  ┌── 시청 이력 (video IDs) → Embedding → Average ──┐
  ├── 검색 이력 (search tokens) → Embedding → Average ┤
  ├── 지역 (geographic) → Embedding ─────────────────┤
  ├── Example Age (학습 데이터 나이) ──────────────────┤
  ├── 성별, 로그인 상태, 나이 등 ──────────────────────┤
  └──────────────── Concatenate ──────────────────────┘
                        ↓
              ReLU (2048) → ReLU (1024) → ReLU (512) → ReLU (256)
                        ↓
              User Embedding (u) ∈ R^256
                        ↓
         [학습 시] Sampled Softmax  |  [서빙 시] ANN 검색
```

### 45.3 핵심 설계 결정들

#### Implicit Feedback 사용

| 선택지 | YouTube의 결정 | 이유 |
|--------|---------------|------|
| Explicit feedback (좋아요/싫어요) | ✗ 사용 안 함 | 데이터가 극히 희소 |
| **Implicit feedback (시청 완료)** | **✓ 사용** | 데이터가 풍부, 긴 꼬리(long tail) 커버 |

#### 가변 길이 입력의 처리

사용자마다 시청 이력 길이가 다르다. YouTube는 **embedding → average** 전략을 사용:
- 각 동영상 ID를 임베딩 (256차원)
- 최근 50개 시청/검색의 임베딩을 **평균** → 고정 크기 벡터

#### Example Age 피처 (핵심 트릭)

학습 데이터는 과거 기반이므로 모델은 **과거 인기 콘텐츠에 편향**된다. 이를 해결하기 위해:
- **학습 시**: 학습 데이터의 나이(`tmax - t`)를 피처로 입력
- **서빙 시**: 이 피처를 0 또는 약간 음수로 설정 → "지금 시점"의 추천

#### Label 선택: 미래 시청 예측

```
❌ 랜덤 hold-out 예측: 미래 정보 누출, 비대칭 소비 패턴 무시
✓ 미래 시청 예측: 시간순으로 이전 행동만 입력, 다음 시청을 예측
```

---

## 46. Ranking — 정밀한 순위 결정

### 46.1 Candidate Generation과의 차이

| 항목 | Candidate Generation | Ranking |
|------|---------------------|---------|
| **입력 규모** | 수백만 | 수백 |
| **피처 수** | 소수 (거친 피처) | **수백 개** (풍부한 피처) |
| **목표** | Broad recall | **Fine-grained precision** |
| **출력** | Top-N 후보 | 각 후보의 **점수** (순위) |
| **최적화 목표** | 시청 확률 | **Expected Watch Time** |

### 46.2 피처 엔지니어링 (논문의 가장 실용적인 부분)

YouTube Ranking에서 사용하는 피처 분류:

| 피처 카테고리 | 예시 | 우리 프로젝트 대응 |
|-------------|------|-----------------|
| **사용자-아이템 상호작용** | 이 채널의 동영상을 몇 개 봤는가, 마지막 시청 시점 | 이 운동 카테고리의 루틴을 몇 번 수행했는가, 마지막 수행 시점 |
| **카테고리 피처** | 동영상 ID, 채널 ID, 검색 쿼리 | 운동 ID, 근육 부위, 난이도 |
| **연속 피처** | 시청 시간, 노출 횟수, 나이 | 수행 시간, 루틴 완료율, 사용자 운동 경력 |
| **노출 피처** | 이 동영상이 몇 번 노출되었지만 클릭되지 않았는가 | 이 운동이 추천되었지만 건너뛰었는가 |
| **후보 생성 피처** | 어떤 소스에서 후보로 선정되었는가, 그 점수는? | 검색 유사도 점수, 리랭킹 점수 |

### 46.3 피처 처리 기법

#### Categorical → Embedding

```
video_id (수백만 종류)  → Embedding(dim=32) → Dense vector
search_query (수백만 종류) → Embedding(dim=32) → Dense vector
```

**Shared Embedding**: 동일 ID 공간(예: video_id)의 여러 피처가 임베딩을 공유하되, 네트워크 상위 레이어에서 각각 다르게 활용

#### Continuous → Normalization + Powers

```
연속 피처 x → CDF 정규화 → x̃ ∈ [0, 1)
네트워크 입력: [x̃, x̃², √x̃]  ← 비선형 관계 학습 용이
```

### 46.4 Expected Watch Time 최적화 (핵심 인사이트)

| 최적화 목표 | 문제점 |
|-----------|--------|
| **Click-Through Rate** | "낚시 영상"(clickbait) 문제 — 클릭은 되지만 곧 이탈 |
| **Expected Watch Time** | ✓ 실제 engagement 반영 — 사용자가 오래 볼수록 가치 높음 |

**구현**: Weighted Logistic Regression
- Positive (클릭): **시청 시간으로 가중치** (오래 시청 = 높은 가중치)
- Negative (미클릭): 가중치 1
- 결과: `odds ≈ E[T]` (기대 시청 시간 근사)

---

## 47. 실험 결과 — 깊이와 피처의 중요성

### 47.1 Candidate Generation: 피처 + 깊이 효과

| 설정 | Holdout MAP |
|------|-------------|
| Watches Only, Depth 0 (선형 모델) | 낮음 (baseline) |
| + Searches | +개선 |
| + Example Age | +개선 |
| **All Features, Depth 4** | **최고 성능** |

→ **피처를 추가할수록, 깊이를 늘릴수록 성능 향상**. 단, Depth 3 이후 수렴.

### 47.2 Ranking: 히든 레이어 효과

| Hidden Layers | Weighted Per-user Loss |
|---------------|----------------------|
| None (선형 모델) | 41.6% |
| 256 ReLU | 36.9% |
| 1024 ReLU | 35.8% |
| **1024→512→256 ReLU** | **34.6%** |

→ 선형 모델 대비 **7%p 개선**. 1024→512→256 "타워" 패턴이 최적.

### 47.3 핵심 실전 교훈

| 교훈 | 설명 | 우리 프로젝트 적용 |
|------|------|-----------------|
| **Offline ≠ Online** | 오프라인 지표가 좋아도 A/B 테스트에서 나쁠 수 있음 | 최종 판단은 사용자 피드백 기반 |
| **사용자 균등 가중** | 소수 활발 사용자가 손실 함수를 지배하지 않도록 | 사용자당 고정 수의 학습 예시 |
| **정보 은닉** | 서빙 시 없는 정보를 학습에 넣으면 과적합 | 순서 정보, 검색 결과 페이지 정보 제거 |
| **Watch Time > CTR** | 클릭률보다 시청 시간이 진짜 engagement | 루틴 완료율 > 루틴 클릭률 |
| **Example Age** | 학습 데이터의 시간 편향 제거 | 운동 트렌드 변화 반영 |

---

## 48. 우리 프로젝트에의 핵심 적용 — YouTube 2단계를 RAG+LLM으로

### 48.1 YouTube vs 우리 프로젝트: 1:1 매핑

| YouTube 컴포넌트 | 우리 프로젝트 대응 | 구현 파일 |
|-----------------|-----------------|----------|
| **Video Corpus** (수백만) | 운동 문서 코퍼스 (수백~수천) | `data/exercises/` |
| **User Watch History** → Embedding | 사용자 운동 이력 → BGE-M3 쿼리 | `retriever.py` |
| **Candidate Generation** (ANN 검색) | FAISS Dense+Sparse 검색 → Top-20 | `retriever.py` |
| **Ranking Model** (피처 기반 스코어링) | bge-reranker-v2-m3 → Top-5 | `reranker.py` |
| **Final Presentation** | Qwen2.5 LLM이 개인화 루틴 생성 | `vllm_client.py` |
| **Implicit Feedback** (시청 완료) | 루틴 완료율, 운동 수행 여부 | `feedback/` |
| **Feature Engineering** | 사용자 프로필 피처 (경력, 목표, 부위) | `prompt_builder.py` |
| **Expected Watch Time** 최적화 | **루틴 완료율 × 사용자 만족도** 최적화 | 평가 메트릭 |

### 48.2 YouTube 피처 설계를 우리 프로젝트에 적용

YouTube의 피처 카테고리를 그대로 가져와 피트니스 도메인에 매핑:

```python
# 사용자 프로필 (YouTube의 Demographics에 해당)
user_profile = {
    "experience": "beginner",       # 운동 경력
    "goal": "muscle_gain",          # 목표 (근력, 다이어트, 건강)
    "available_time": 45,           # 가용 시간 (분)
    "equipment": ["barbell", "dumbbell"],  # 보유 장비
    "injuries": ["lower_back"],     # 부상 이력
    "age": 28,
    "gender": "male"
}

# 사용자-운동 상호작용 이력 (YouTube의 Watch History에 해당)
user_history = {
    "completed_routines": [         # 최근 완료한 루틴
        {"name": "가슴 Day A", "completed_at": "2026-03-14", "completion_rate": 1.0},
        {"name": "등 Day B", "completed_at": "2026-03-13", "completion_rate": 0.8},
    ],
    "skipped_exercises": ["바벨 스쿼트"],  # 건너뛴 운동 (YouTube의 미클릭)
    "favorite_exercises": ["벤치프레스", "인클라인 덤벨"],  # 자주 수행한 운동
    "muscle_groups_recent": {       # 최근 근육 부위별 빈도
        "chest": 3, "back": 2, "legs": 0  # ← legs가 0이면 하체 운동 추천 우선
    }
}

# Prompt Builder에서 이 정보를 프롬프트에 반영
system_prompt = f"""당신은 전문 피트니스 코치입니다.

사용자 정보:
- 경력: {user_profile['experience']}
- 목표: {user_profile['goal']}
- 가용 시간: {user_profile['available_time']}분
- 부상 이력: {user_profile['injuries']}
- 최근 하체 운동 부족 (최근 {user_history['muscle_groups_recent']['legs']}회)
- 건너뛴 운동: {user_history['skipped_exercises']}

위 정보를 고려하여 운동 루틴을 추천하세요.
건너뛴 운동은 다른 대안으로 대체하세요."""
```

### 48.3 Implicit Feedback 수집 설계

YouTube의 핵심 교훈: **명시적 피드백(좋아요)보다 암묵적 피드백(시청 완료)이 훨씬 풍부하고 유용하다.**

| 수집 신호 | YouTube 대응 | 우리 프로젝트 | 의미 |
|----------|-------------|-------------|------|
| **Positive (강)** | 동영상 끝까지 시청 | 루틴 100% 완료 | 이 루틴이 적합했음 |
| **Positive (약)** | 동영상 일부 시청 | 루틴 50~99% 완료 | 부분적으로 적합 |
| **Negative (약)** | 노출됐지만 미클릭 | 추천됐지만 건너뜀 | 관심 없음 |
| **Negative (강)** | 클릭 후 즉시 이탈 | 루틴 시작 후 <10%에서 포기 | 부적합 |
| **Explicit** | 좋아요/싫어요 | 별점/피드백 | 희소하지만 명확 |

### 48.4 "Example Age" → 운동 트렌드 반영

YouTube의 Example Age 기법을 피트니스에 적용:

- **문제**: 오래된 인기 운동(벤치프레스)에 편향, 최신 트렌드(케틀벨 플로우 등) 미반영
- **해결**: 문서에 **업로드/업데이트 시점**을 메타데이터로 저장, 검색 시 최신 문서에 가산점

```python
# 검색 점수에 시간 가중치 적용
import math
from datetime import datetime

def time_decay_boost(doc_date: str, decay_days: int = 180) -> float:
    """최신 문서에 가산점 부여 (YouTube의 Example Age 아이디어)"""
    age_days = (datetime.now() - datetime.fromisoformat(doc_date)).days
    return math.exp(-age_days / decay_days)  # 180일 반감기

final_score = retrieval_score * 0.8 + time_decay_boost(doc.updated_at) * 0.2
```

---

## 49. YouTube 논문의 한계와 우리 프로젝트에서의 보완

| YouTube 논문의 특성 | 우리 프로젝트와의 차이 | 보완 방법 |
|-------------------|-------------------|----------|
| **대규모** (10억 사용자, 수백만 동영상) | 소규모 (수백~수천 운동) | 2단계 중 1단계를 RAG 검색으로 단순화 |
| **ID 기반 임베딩** (각 동영상의 고유 벡터 학습) | 텍스트 기반 의미 임베딩 (BGE-M3) | 사전학습 임베딩 모델 활용 (학습 데이터 불필요) |
| **DNN 기반 Ranking** | LLM 기반 생성 | Reranker + LLM으로 대체 |
| **CTR/Watch Time 지표** | 루틴 완료율/만족도 | Pydantic 검증 + 사용자 피드백 |
| **Explicit 피처 엔지니어링** 필요 | 프롬프트에 자연어로 피처 전달 | LLM이 피처 해석을 자동 수행 |
| **Cold Start** 문제 (신규 사용자) | 동일 | 사용자 프로필 설문으로 보완 |

→ YouTube의 2단계 아키텍처와 피처 설계 철학은 우리 프로젝트에 그대로 적용 가능하지만, **DNN 대신 RAG+LLM**으로 구현하는 것이 핵심 차이점이다.

---

# 실전 보강: OpenAI Cookbook 기반 프로덕션 RAG 패턴

> **출처**: [OpenAI Cookbook](https://github.com/openai/openai-cookbook) — 실제 LLM 시스템 구성 사례 모음
> **분석 목적**: 논문 기반 아키텍처(섹션 1~42)에서 **실전에서 누락되기 쉬운 패턴**을 식별하고 보강

---

## 50. [보강] Chunking 전략 — 기존 "100-word" 접근의 한계

### 43.1 기존 RESEARCH.md의 문제점

기존 섹션 2에서 RAG 논문의 "Wikipedia를 100-word 청크로 분할"을 그대로 참조했으나, 이는 **Wikipedia 영문 텍스트에 맞춘 설정**이다. 우리 프로젝트(한국어 피트니스 문서)에는 직접 적용하기 어렵다.

### 43.2 OpenAI Cookbook의 Chunking 패턴 (실전)

Cookbook의 `Embedding_Wikipedia_articles_for_search.ipynb`는 다음과 같은 **실전 chunking 전략**을 사용한다:

| 단계 | 방법 | 핵심 포인트 |
|------|------|------------|
| **1. 구조적 분할** | 문서의 헤더/섹션 경계를 기준으로 1차 분할 | 무관한 주제가 섞이지 않도록 |
| **2. 토큰 기반 제한** | 각 섹션을 **최대 1,600 토큰**으로 제한 | 고정 word 수 대신 토큰 수 사용 |
| **3. 재귀적 분할** | 긴 섹션은 `\n\n` → `\n` → `. ` 순서로 의미 경계를 찾아 반으로 나눔 | 문장 중간에서 잘리지 않도록 |
| **4. 제목 보존** | 각 청크에 **부모 제목(H1, H2...)을 앞에 붙임** | 컨텍스트 유실 방지 |
| **5. 노이즈 제거** | References, External links 등 무관한 섹션 제거 | 검색 정확도 향상 |

### 43.3 우리 프로젝트에 적용할 Chunking 전략

```
원본 문서: "벤치프레스 완벽 가이드"
    ↓
1단계: 한국어 섹션 분할 (##, ### 기준)
    → [개요, 자세, 호흡법, 세트/반복, 주의사항, ...]
    ↓
2단계: 제목 prepend
    → "벤치프레스 완벽 가이드 > 자세\n\n바벨을 어깨너비로..."
    ↓
3단계: 토큰 수 체크 (max_tokens=1000)
    → 초과 시 \n\n → \n → . 순서로 재귀 분할
    ↓
4단계: 최종 청크 리스트
```

**핵심 변경점** (기존 "100-word" 대비):
- **토큰 기반**: 한국어는 1 단어 ≈ 2~3 토큰이므로, word 수가 아니라 **토큰 수**로 제한
- **제목 보존**: 검색 시 컨텍스트가 유실되지 않도록 상위 제목을 항상 포함
- **재귀적 분할**: 의미 경계(문단 → 줄바꿈 → 마침표)를 존중하여 분할

---

## 51. [보강] Token Budget 관리 — 프롬프트에 얼마나 넣을 것인가

### 51.1 기존 RESEARCH.md에 부재했던 내용

기존에는 "검색된 문서를 프롬프트에 concatenate"(섹션 2)까지만 언급했고, **실제로 몇 개의 문서를 얼마나 넣을지**에 대한 전략이 없었다.

### 51.2 OpenAI Cookbook의 Token Budget 패턴

```python
def query_message(query, df, model, token_budget):
    """검색된 문서를 token_budget 내에서 최대한 많이 프롬프트에 삽입"""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = '아래 참고 문서를 기반으로 질문에 답하세요...'
    question = f"\n\n질문: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\n참고 문서:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question) > token_budget:
            break
        else:
            message += next_article
    return message + question
```

핵심 원칙:
1. **고정 개수(Top-5)가 아니라 token_budget으로 동적 결정**
2. 관련도 순으로 정렬 후, 예산이 허용하는 만큼 삽입
3. `token_budget = max_context_length - 시스템프롬프트 - 예상응답길이`

### 51.3 우리 프로젝트 적용

| 항목 | 값 | 근거 |
|------|-----|------|
| Qwen2.5 max_model_len | 4096~8192 | 프로덕션 안정성 |
| 시스템 프롬프트 | ~200 토큰 | 피트니스 코치 역할 정의 |
| 예상 응답 길이 | ~1024 토큰 | max_tokens 설정 |
| **문서 삽입 budget** | **~3000~7000 토큰** | 나머지 전부 |
| 청크 크기 (max) | 1000 토큰 | 위의 43.3 |
| → 삽입 가능 문서 수 | **~3~7개** | budget ÷ 청크 크기 |

→ Reranker 상위 결과를 token budget 내에서 **동적으로** 삽입. 고정 Top-K가 아님.

---

## 52. [보강] Guardrails — 입력/출력 안전장치

### 52.1 기존 RESEARCH.md에 부재했던 내용

기존 "출력 검증"(섹션 43 매핑 테이블의 `validation/`)은 **Pydantic 스키마 검증**만 언급했다. 이것은 **구조적 검증**(JSON 형식)이지, **의미적 검증**(할루시네이션, 부적절 응답)이 아니다.

### 52.2 OpenAI Cookbook의 Guardrails 아키텍처

Cookbook은 **입력 가드레일**과 **출력 가드레일** 두 계층을 제시한다:

```
사용자 입력
    ↓
┌── Input Guardrail (비동기 병렬) ──┐
│  • 주제 범위 체크: 피트니스 관련? │
│  • 프롬프트 인젝션 탐지           │
│  • 부적절 컨텐츠 필터             │
└──────────────────────────────────┘
    ↓ (allowed인 경우만 통과)
RAG 파이프라인 (검색 → 리랭킹 → LLM)
    ↓
┌── Output Guardrail ──────────────┐
│  • Pydantic 스키마 검증 (구조적)  │
│  • 할루시네이션 체크 (의미적)     │
│  • 위험 운동 추천 필터 (도메인)   │
└──────────────────────────────────┘
    ↓
최종 응답
```

### 52.3 비동기 실행 패턴 (핵심)

Cookbook의 가장 중요한 패턴: **가드레일과 LLM 호출을 비동기 병렬 실행**하여 레이턴시 최소화

```python
async def execute_with_guardrails(user_request):
    # 입력 가드레일과 RAG 파이프라인을 동시 실행
    guardrail_task = asyncio.create_task(input_guardrail(user_request))
    rag_task = asyncio.create_task(rag_pipeline(user_request))

    done, _ = await asyncio.wait(
        [guardrail_task, rag_task],
        return_when=asyncio.FIRST_COMPLETED
    )

    if guardrail_task in done:
        if guardrail_task.result() == "not_allowed":
            rag_task.cancel()  # 차단 시 LLM 호출 취소 → 비용 절약
            return "죄송합니다, 피트니스 관련 질문만 답변할 수 있습니다."

    rag_response = await rag_task
    # 출력 가드레일
    moderation = await output_guardrail(rag_response)
    if moderation.score >= 3:
        return "안전한 응답을 생성할 수 없습니다. 다시 시도해주세요."
    return rag_response
```

### 52.4 우리 프로젝트에 필요한 가드레일

| 가드레일 | 유형 | 목적 | 구현 방식 |
|----------|------|------|----------|
| **주제 범위** | Input | 피트니스 무관 질문 차단 | LLM 기반 분류 or 키워드 |
| **Pydantic 검증** | Output | JSON 스키마 준수 확인 | Pydantic 모델 |
| **할루시네이션 체크** | Output | 검색 문서에 없는 운동 추천 방지 | 참고 문서 vs 응답 교차 검증 |
| **위험 운동 필터** | Output | 초보자에게 위험한 운동 경고 | 규칙 기반 + LLM 판단 |

---

## 53. [보강] 평가(Eval) 기반 개발 — "테스트 없는 RAG는 프로토타입"

### 53.1 기존 RESEARCH.md에 부재했던 내용

논문 분석 중심으로 작성되어, **실제 시스템의 평가 방법론**이 빠져 있었다.

### 53.2 OpenAI의 평가 프레임워크

OpenAI는 RAG 시스템 평가에 대해 다음을 권장한다:

#### 평가 대상 3계층

| 계층 | 측정 대상 | 메트릭 |
|------|----------|--------|
| **Retrieval** | 검색이 관련 문서를 찾았는가 | Recall@K, MRR, nDCG |
| **Generation** | LLM 응답이 정확한가 | ROUGE, BERTScore, 인간 평가 |
| **End-to-End** | 전체 시스템이 올바른 답을 내는가 | Accuracy, Faithfulness (할루시네이션 비율) |

#### Eval-Driven Development 원칙

1. **평가 세트 먼저 구축**: 구현 전에 Q&A 쌍을 만든다
2. **자동화된 평가**: 모든 변경에 대해 eval 실행
3. **연속적 평가**: eval set을 계속 확장
4. **로깅**: 프로덕션 질의/응답을 기록하여 eval 데이터로 활용

### 53.3 우리 프로젝트 평가 계획

```
평가 세트 구축:
  질문: "초보자 가슴 운동 추천해줘"
  기대 검색 문서: [벤치프레스, 인클라인 덤벨, 푸시업, ...]
  기대 응답 포함: ["세트 수", "반복 수", "휴식 시간"]
  금지 응답: ["데드리프트", "스쿼트"] (가슴 운동이 아님)

평가 메트릭:
  1. Retrieval Recall@5: 기대 문서 중 검색된 비율
  2. Faithfulness: 응답이 검색 문서 내용만 사용했는가
  3. Schema compliance: JSON 스키마를 준수했는가
  4. Safety: 위험 운동 추천이 없는가
```

---

## 54. [보강] Structured Outputs — JSON 형식 보장

### 54.1 기존 RESEARCH.md의 접근 vs 실전 패턴

기존에는 "Pydantic 스키마 검증"을 출력 후 사후 검증으로만 언급했다. 하지만 실전에서는 **LLM에게 출력 형식을 강제**하는 것이 더 효과적이다.

### 54.2 Structured Outputs 방식

OpenAI의 Structured Outputs 기능은 `response_format`으로 **JSON Schema를 강제**한다. 오픈소스 LLM(Qwen2.5)에서는 동일한 효과를 위해:

#### 방법 1: 프롬프트에 JSON 스키마 명시

```python
system_prompt = """당신은 전문 피트니스 코치입니다.
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.

{
  "routine_name": "루틴 이름",
  "difficulty": "beginner|intermediate|advanced",
  "exercises": [
    {
      "name": "운동명",
      "sets": 3,
      "reps": "8-12",
      "rest_seconds": 60,
      "notes": "주의사항"
    }
  ],
  "total_time_minutes": 45,
  "warnings": ["해당되는 경우 안전 주의사항"]
}"""
```

#### 방법 2: vLLM의 guided decoding (JSON Schema 강제)

```bash
# vLLM은 --guided-decoding-backend 옵션으로 JSON Schema 강제 지원
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --guided-decoding-backend outlines
```

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=messages,
    extra_body={
        "guided_json": exercise_routine_schema  # JSON Schema 직접 전달
    }
)
```

#### 방법 3: Pydantic으로 사후 검증 (기존 계획 + 재시도)

```python
from pydantic import BaseModel, validator
from typing import List

class Exercise(BaseModel):
    name: str
    sets: int
    reps: str
    rest_seconds: int
    notes: str = ""

class Routine(BaseModel):
    routine_name: str
    difficulty: str
    exercises: List[Exercise]
    total_time_minutes: int
    warnings: List[str] = []

# 파싱 실패 시 재시도 로직
for attempt in range(3):
    raw = llm_generate(prompt)
    try:
        routine = Routine.model_validate_json(raw)
        break
    except ValidationError:
        prompt += "\n\n이전 응답이 JSON 형식에 맞지 않았습니다. 다시 시도해주세요."
```

→ **방법 1(프롬프트) + 방법 3(Pydantic 사후 검증)을 기본으로**, 정확도 문제 시 방법 2(guided decoding) 도입.

---

# Pydantic — JSON 스키마 검증이 왜 필요한가

> **출처**: [Pydantic 공식 문서](https://docs.pydantic.dev/latest/)
> **버전**: v2.12+ (Rust 기반 pydantic-core, jiter JSON 파서)
> **라이선스**: MIT

---

## 55. LLM 출력에 JSON 스키마 검증이 필요한 이유

### 55.1 핵심 문제: LLM은 "구조화된 데이터"를 보장하지 않는다

LLM은 자연어 텍스트를 생성하는 모델이다. JSON을 요청해도 다음과 같은 실패가 빈번하게 발생한다:

| 실패 유형 | 예시 | 빈도 |
|-----------|------|------|
| **필드명 오류** | `"exercise_name"` 대신 `"name_of_exercise"` | 매우 흔함 |
| **필수 필드 누락** | `exercises` 배열 없이 텍스트로만 응답 | 흔함 |
| **타입 오류** | `"sets": "세 세트"` (int여야 하지만 str 반환) | 흔함 |
| **JSON 구문 오류** | 쉼표 누락, 따옴표 불일치, 후행 쉼표 | 간헐적 |
| **추가 텍스트** | `"다음은 추천 루틴입니다:\n{...}"` (JSON 앞에 텍스트) | 흔함 |
| **불완전 JSON** | 토큰 제한으로 중간에 잘림 | max_tokens 도달 시 |

연구에 따르면, 구조화 출력 강제 없이 GPT-4급 모델도 **복잡한 JSON 스키마 준수율이 40% 미만**이다. 검증 없이 이 출력을 파싱하면 **런타임 에러 → 사용자에게 빈 응답 또는 크래시**로 이어진다.

### 55.2 검증이 없으면 일어나는 일 (우리 프로젝트 맥락)

```python
# ❌ 검증 없는 코드
response = llm_generate(prompt)
data = json.loads(response)  # JSONDecodeError 가능
exercises = data["exercises"]  # KeyError 가능
for ex in exercises:
    print(f"{ex['name']}: {ex['sets']}세트")  # KeyError, TypeError 가능
```

위 코드의 **모든 줄에서 에러가 발생할 수 있다**. 실제 프로덕션에서는:
1. 사용자가 운동 추천을 요청했는데 빈 화면만 보임
2. 서버 로그에 500 에러가 쌓임
3. 원인 파악이 어려움 (LLM 출력이 매번 다르므로)

### 55.3 Pydantic이 해결하는 것

[Pydantic](https://docs.pydantic.dev/latest/)은 Python 타입 힌트를 기반으로 **런타임 데이터 검증과 직렬화**를 수행하는 라이브러리다.

```
LLM 출력 (문자열)
    ↓
Pydantic model_validate_json()
    ↓
┌─ 성공: 타입 안전한 Python 객체 반환 ──→ 앱 로직에서 안전하게 사용
└─ 실패: ValidationError (어디서, 왜 실패했는지 상세 보고) ──→ 재시도 또는 에러 처리
```

---

## 56. Pydantic 핵심 개념 — 모델, 검증, JSON Schema

### 56.1 BaseModel — 스키마를 Python 클래스로 정의

```python
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum


class Difficulty(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class Exercise(BaseModel):
    name: str = Field(description="운동 이름")
    sets: int = Field(ge=1, le=10, description="세트 수")
    reps: str = Field(description="반복 수 (예: '8-12')")
    rest_seconds: int = Field(ge=0, le=300, description="세트 간 휴식(초)")
    notes: str = ""


class Routine(BaseModel):
    routine_name: str
    difficulty: Difficulty
    exercises: List[Exercise] = Field(min_length=1, max_length=10)
    total_time_minutes: int = Field(ge=10, le=120)
    warnings: List[str] = []
```

이 정의 하나로 Pydantic이 자동으로 제공하는 것:

| 기능 | 설명 |
|------|------|
| **타입 검증** | `sets`가 int인지, `difficulty`가 enum 값인지 체크 |
| **범위 제약** | `sets`는 1~10, `rest_seconds`는 0~300 |
| **필수/선택 필드** | `notes`는 기본값 있어 선택, `name`은 필수 |
| **중첩 구조** | `exercises`는 `Exercise` 모델의 리스트 |
| **리스트 길이** | `exercises`는 1~10개 |
| **타입 강제 변환** | `"3"` → `3` (str → int 자동 변환) |
| **JSON Schema 생성** | `Routine.model_json_schema()` → JSON Schema dict |
| **상세 에러 메시지** | 어떤 필드에서, 왜 실패했는지 정확히 보고 |

### 56.2 model_validate_json — LLM 출력을 직접 검증

```python
from pydantic import ValidationError

llm_output = '{"routine_name": "초보자 가슴", "difficulty": "beginner", ...}'

try:
    routine = Routine.model_validate_json(llm_output)
    # ✓ 타입 안전한 객체. 모든 필드가 정확한 타입으로 보장됨
    print(routine.exercises[0].name)  # 절대 KeyError 없음
    print(routine.exercises[0].sets)  # 반드시 int, 1~10 범위
except ValidationError as e:
    # 상세한 에러 정보 → 재시도 프롬프트에 활용 가능
    print(e.errors())
    # [{'type': 'missing', 'loc': ('exercises',), 'msg': 'Field required', ...}]
```

**`json.loads()` vs `model_validate_json()` 비교**:

| 항목 | `json.loads()` | `Routine.model_validate_json()` |
|------|---------------|-------------------------------|
| JSON 구문 검사 | ✓ | ✓ |
| 필드 존재 확인 | ✗ | ✓ |
| 타입 검사 | ✗ | ✓ |
| 범위/제약 검사 | ✗ | ✓ |
| 자동 타입 변환 | ✗ | ✓ (str→int 등) |
| 에러 위치 보고 | 줄/컬럼만 | 필드 경로 (e.g. `exercises.0.sets`) |
| 반환 타입 | `dict` (타입 불안전) | `Routine` 객체 (타입 안전) |

### 56.3 Partial JSON Parsing — 불완전 JSON 처리

LLM이 `max_tokens` 제한에 도달하면 JSON이 잘린 채로 반환된다. Pydantic v2.7+은 **Partial JSON Parsing**을 지원한다:

```python
from pydantic_core import from_json
from pydantic import BaseModel
from typing import List, Optional


class Exercise(BaseModel):
    name: str
    sets: int = 3
    reps: str = "8-12"


class Routine(BaseModel):
    routine_name: str = ""
    exercises: List[Exercise] = []


# LLM 출력이 중간에 잘린 경우
truncated = '{"routine_name": "가슴 루틴", "exercises": [{"name": "벤치프레스", "sets": 3, "reps": "8-12"}, {"name": "인클라'

partial_data = from_json(truncated, allow_partial=True)
# → {'routine_name': '가슴 루틴', 'exercises': [{'name': '벤치프레스', 'sets': 3, 'reps': '8-12'}]}
# 불완전한 두 번째 운동은 무시, 첫 번째는 정상 파싱

routine = Routine.model_validate(partial_data)
print(routine.exercises[0].name)  # "벤치프레스" — 안전하게 사용 가능
```

→ `max_tokens` 도달로 잘린 JSON에서도 **파싱 가능한 부분만 추출**하여 최대한 활용.

### 56.4 model_json_schema — LLM에게 스키마를 알려주기

Pydantic 모델에서 **JSON Schema를 자동 생성**하여 프롬프트에 포함할 수 있다:

```python
import json

schema = Routine.model_json_schema()
print(json.dumps(schema, indent=2, ensure_ascii=False))
```

출력:
```json
{
  "$defs": {
    "Difficulty": {
      "enum": ["beginner", "intermediate", "advanced"],
      "type": "string"
    },
    "Exercise": {
      "properties": {
        "name": {"description": "운동 이름", "type": "string"},
        "sets": {"description": "세트 수", "maximum": 10, "minimum": 1, "type": "integer"},
        "reps": {"description": "반복 수 (예: '8-12')", "type": "string"},
        "rest_seconds": {"description": "세트 간 휴식(초)", "maximum": 300, "minimum": 0, "type": "integer"},
        "notes": {"default": "", "type": "string"}
      },
      "required": ["name", "sets", "reps", "rest_seconds"],
      "type": "object"
    }
  },
  "properties": {
    "routine_name": {"type": "string"},
    "difficulty": {"$ref": "#/$defs/Difficulty"},
    "exercises": {"items": {"$ref": "#/$defs/Exercise"}, "maxItems": 10, "minItems": 1, "type": "array"},
    "total_time_minutes": {"maximum": 120, "minimum": 10, "type": "integer"},
    "warnings": {"default": [], "items": {"type": "string"}, "type": "array"}
  },
  "required": ["routine_name", "difficulty", "exercises", "total_time_minutes"],
  "type": "object"
}
```

이 스키마를 **프롬프트에 직접 포함**하거나, **vLLM의 guided_json에 전달**할 수 있다.

---

## 57. Custom Validators — 도메인 특화 검증

### 57.1 왜 타입 검증만으로는 부족한가

타입 검증은 "sets가 int인가?"만 확인한다. 하지만 우리 프로젝트에서는 **도메인 로직 검증**이 필요하다:

- 초보자 루틴에 "데드리프트 1RM 테스트"가 포함되면 안 됨
- 총 시간이 exercises의 세트 수 × 휴식 시간보다 짧으면 비논리적
- "벤치프레스: 100세트"는 int이지만 비현실적

### 57.2 Field Validator — 필드 단위 검증

```python
from pydantic import BaseModel, Field, field_validator


class Exercise(BaseModel):
    name: str
    sets: int = Field(ge=1, le=10)
    reps: str
    rest_seconds: int = Field(ge=0, le=300)

    @field_validator('reps')
    @classmethod
    def validate_reps_format(cls, v: str) -> str:
        """reps가 '8-12' 또는 '15' 형식인지 확인"""
        import re
        if not re.match(r'^\d+(-\d+)?$', v):
            raise ValueError(f"reps 형식이 올바르지 않습니다: '{v}'. '8-12' 또는 '15' 형식이어야 합니다.")
        return v
```

→ `"reps": "많이 하세요"` 같은 LLM의 자유형 응답을 잡아낸다.

### 57.3 Model Validator — 모델 전체의 교차 검증

```python
from pydantic import model_validator


class Routine(BaseModel):
    routine_name: str
    difficulty: Difficulty
    exercises: List[Exercise] = Field(min_length=1, max_length=10)
    total_time_minutes: int = Field(ge=10, le=120)
    warnings: List[str] = []

    @model_validator(mode='after')
    def validate_routine_consistency(self):
        """루틴 전체의 논리적 일관성 검증"""
        # 1. 최소 예상 시간 계산
        min_time = sum(
            ex.sets * (30 + ex.rest_seconds)  # 각 세트 최소 30초 + 휴식
            for ex in self.exercises
        ) // 60

        if self.total_time_minutes < min_time:
            raise ValueError(
                f"총 시간({self.total_time_minutes}분)이 운동 최소 소요 시간({min_time}분)보다 짧습니다."
            )

        # 2. 초보자 위험 운동 체크
        dangerous_for_beginners = {"바벨 스쿼트", "데드리프트", "클린 앤 저크", "스내치"}
        if self.difficulty == Difficulty.beginner:
            for ex in self.exercises:
                if ex.name in dangerous_for_beginners:
                    self.warnings.append(
                        f"⚠️ '{ex.name}'은 초보자에게 위험할 수 있습니다. 전문가 지도 하에 수행하세요."
                    )

        return self
```

→ **필드 간 교차 검증**: 개별 필드는 유효하지만 **조합이 비논리적인 경우**를 잡아낸다.

---

## 58. 전체 검증 파이프라인 — Pydantic이 프로젝트에서 담당하는 역할

```
LLM 출력 (raw string)
    ↓
┌── 1단계: JSON 파싱 ──────────────────────┐
│  model_validate_json() 또는              │
│  from_json(allow_partial=True) 후        │
│  model_validate()                         │
│                                           │
│  → JSON 구문 오류 → 재시도 (최대 3회)    │
└───────────────────────────────────────────┘
    ↓ (JSON 파싱 성공)
┌── 2단계: 타입 + 제약 검증 ───────────────┐
│  Pydantic 자동 수행:                     │
│  • 필수 필드 존재 확인                   │
│  • 타입 검사 (str, int, List, Enum)      │
│  • 범위 제약 (ge, le, min_length 등)     │
│  • 자동 타입 변환 ("3" → 3)             │
│                                           │
│  → ValidationError → 에러 메시지를       │
│    프롬프트에 추가하여 재시도             │
└───────────────────────────────────────────┘
    ↓ (타입 검증 통과)
┌── 3단계: 도메인 로직 검증 ───────────────┐
│  Custom Validators 수행:                 │
│  • reps 형식 검증 ('8-12' 패턴)          │
│  • 총 시간 vs 실제 소요 시간 일관성      │
│  • 초보자 위험 운동 경고                 │
│                                           │
│  → ValidationError → 재시도 또는 경고    │
└───────────────────────────────────────────┘
    ↓ (모든 검증 통과)
✓ Routine 객체 — 타입 안전, 도메인 유효, 바로 사용 가능
```

### 58.1 검증 실패 시 재시도 전략 (상세)

```python
from pydantic import ValidationError

def generate_validated_routine(query: str, context_docs: list, max_retries: int = 3) -> Routine:
    prompt = build_prompt(query, context_docs)

    for attempt in range(max_retries):
        raw_output = llm_generate(prompt)

        try:
            routine = Routine.model_validate_json(raw_output)
            return routine

        except ValidationError as e:
            # Pydantic 에러 메시지를 다음 시도의 프롬프트에 포함
            error_details = []
            for err in e.errors():
                loc = " → ".join(str(l) for l in err['loc'])
                error_details.append(f"  - {loc}: {err['msg']}")

            error_feedback = "\n".join(error_details)
            prompt += f"\n\n이전 응답에서 다음 오류가 발생했습니다:\n{error_feedback}\n위 오류를 수정하여 다시 JSON을 생성하세요."

    # 모든 재시도 실패
    return None
```

핵심: Pydantic의 **상세한 에러 메시지**(`loc`, `msg`)를 LLM에게 피드백으로 전달하면, 다음 시도에서 정확도가 크게 향상된다.

---

## 59. Pydantic이 우리 프로젝트에 적합한 이유 — 요약

| 요구사항 | Pydantic 대응 |
|----------|--------------|
| **LLM 출력 파싱** | `model_validate_json()` — JSON 문자열을 직접 검증+파싱 |
| **타입 안전성** | Python 타입 힌트 기반 — IDE 자동완성, 타입 체크 |
| **범위 제약** | `Field(ge=1, le=10)` — 비현실적 값 차단 |
| **enum 제약** | `Difficulty(str, Enum)` — 허용된 값만 통과 |
| **중첩 구조** | `List[Exercise]` — 복잡한 JSON 구조 검증 |
| **도메인 로직** | `@field_validator`, `@model_validator` — 커스텀 비즈니스 규칙 |
| **불완전 JSON** | `from_json(allow_partial=True)` — 잘린 출력도 최대한 활용 |
| **JSON Schema 생성** | `model_json_schema()` — 프롬프트/guided decoding에 활용 |
| **에러 피드백** | `e.errors()` — LLM 재시도에 상세 에러 정보 전달 |
| **성능** | Rust 기반 pydantic-core — 일반 Python 검증 대비 ~17배 빠름 |
| **생태계** | FastAPI, LangChain, OpenAI SDK 등 주요 프레임워크 통합 |

→ Pydantic은 단순한 "사후 검증 도구"가 아니라, **LLM 출력의 신뢰성을 보장하는 핵심 인프라**다. 스키마 정의 → JSON Schema 생성(프롬프트/guided decoding) → 런타임 검증 → 에러 피드백 재시도까지 **전체 구조화 출력 파이프라인을 하나의 모델 정의로 통합**한다.

---

## 60. [보강] 검색 → 질의 흐름에서 누락된 "답변 불가" 처리

### 60.1 Cookbook의 핵심 패턴: "I don't know"

OpenAI Cookbook의 QA 예제에서 가장 강조하는 패턴:

```
프롬프트: "아래 참고 문서를 기반으로 질문에 답하세요.
          답을 찾을 수 없으면 '답변할 수 없습니다'라고 말하세요."
```

이것이 **할루시네이션을 가장 효과적으로 방지**하는 방법이다. 검색된 문서에 관련 정보가 없을 때 LLM이 "모르겠다"고 말할 수 있게 명시적으로 허용해야 한다.

### 60.2 기존 프롬프트 설계의 문제

섹션 42의 프롬프트 흐름도에서:
```
[System] 당신은 전문 피트니스 코치입니다.
         아래 참고 문서를 기반으로 운동 루틴을 JSON 형식으로 추천하세요.
```

→ **"답변할 수 없을 때 어떻게 해야 하는가"가 빠져 있다**. LLM은 지시에 따라 무조건 답변을 생성하려 하며, 이때 할루시네이션이 발생한다.

### 60.3 수정된 프롬프트 전략

```
[System] 당신은 전문 피트니스 코치입니다.
         아래 참고 문서를 기반으로 운동 루틴을 JSON 형식으로 추천하세요.
         
         중요 규칙:
         1. 반드시 참고 문서에 있는 정보만 사용하세요.
         2. 참고 문서에서 답을 찾을 수 없으면 {"error": "관련 정보를 찾을 수 없습니다"}를 반환하세요.
         3. 확실하지 않은 정보는 추측하지 마세요.
```

---

## 61. [보강] 검색 디버깅 — 실패 원인 분류

### 61.1 Cookbook의 트러블슈팅 프레임워크

OpenAI Cookbook은 RAG 시스템의 오답을 두 가지 원인으로 분류한다:

| 실패 유형 | 원인 | 디버깅 방법 | 해결 방향 |
|-----------|------|------------|----------|
| **검색 실패** | 관련 문서를 찾지 못함 | `print_message=True`로 LLM에 전달된 문서 확인 | 청킹 전략, 임베딩 모델, 인덱스 파라미터 조정 |
| **생성 실패** | 문서는 찾았지만 LLM이 잘못 해석 | 전달된 문서에 답이 있는지 수동 확인 | 프롬프트 개선, temperature 낮춤, few-shot 예시 |

### 61.2 우리 프로젝트에서의 디버깅 파이프라인

```python
def ask_with_debug(query, print_context=False):
    # 1. 검색
    retrieved_docs = retrieve(query, top_k=20)
    reranked_docs = rerank(query, retrieved_docs, top_k=5)

    if print_context:
        print("=== 검색된 문서 ===")
        for doc in reranked_docs:
            print(f"  Score: {doc.score:.3f} | {doc.text[:100]}...")

    # 2. 프롬프트 구성 (token budget)
    prompt = build_prompt(query, reranked_docs, token_budget=3000)

    if print_context:
        print(f"\n=== 프롬프트 토큰 수: {count_tokens(prompt)} ===")

    # 3. LLM 생성
    response = generate(prompt)

    # 4. 검증
    try:
        routine = Routine.model_validate_json(response)
    except ValidationError as e:
        print(f"검증 실패: {e}")
        return None

    return routine
```

이 패턴으로 "문제가 검색에 있는지, 생성에 있는지"를 빠르게 분리할 수 있다.

---

## 62. [보강] HyDE (Hypothetical Document Embeddings) — 검색 품질 향상 기법

### 62.1 Cookbook이 언급한 고급 검색 기법

Cookbook의 검색 섹션에서 다음을 권장한다:

> "질문은 종종 정답과 **어휘적으로 겹치지 않는다**. 이런 경우 HyDE를 사용하면 검색 품질이 향상된다."

HyDE 원리:
```
사용자 질문: "초보자 가슴 운동 추천해줘"
    ↓
LLM이 가상 답변 생성: "초보자에게 적합한 가슴 운동으로는 벤치프레스 3세트..."
    ↓
가상 답변을 임베딩하여 검색 (질문 대신)
    ↓
문서와 더 잘 매칭됨 (질문보다 답변이 문서와 어휘적으로 유사하므로)
```

### 62.2 프로젝트 적용 판단

- **장점**: 검색 recall 향상, 특히 한국어 질문 ↔ 문서 간 어휘 불일치 시 효과적
- **단점**: LLM 호출 1회 추가 → 레이턴시/비용 증가
- **결정**: 초기에는 미적용, Retrieval Recall이 낮을 경우 도입 검토

---

## 63. 전체 통합 아키텍처 매핑 (최종)

| 파이프라인 단계 | 이론적 근거 | 우리의 선택 | Cookbook 보강 | 프로젝트 파일 |
|---|---|---|---|---|
| **문서 청킹** | RAG(100-word) | 구조적 분할 + 토큰 기반(1000 토큰) | **재귀적 분할, 제목 보존** (§50) | `chunker.py` |
| 문서 인코딩 | RAG(BERT_d) + SBERT(Bi-encoder) | **BGE-M3** (dense+sparse+colbert) | — | `embedder.py` |
| 벡터 저장 | FAISS(IVFADC/PQ) | **FAISS IndexFlatIP(1024)** | — | `vector_store.py` |
| 유사도 검색 | RAG(Top-K MIPS) + FAISS(WarpSelect) + **YouTube 2단계(§44)** | **Dense+Sparse hybrid search** | HyDE 검토 (§62) | `retriever.py` |
| Re-ranking | SBERT(Cross-encoder > Bi-encoder) + **YouTube Ranking(§46)** | **bge-reranker-v2-m3** | — | `reranker.py` |
| **프롬프트 구성** | RAG(concatenate x,z) + **YouTube 피처 설계(§48)** | 시스템 프롬프트 + 사용자 프로필 + 검색 문서 + 질의 | **Token budget 동적 삽입** (§51), **"답변 불가" 처리** (§60) | `prompt_builder.py` |
| LLM 서빙 | vLLM(PagedAttention, KV Cache) | **vLLM + Qwen2.5-7B-Instruct** | — | `vllm_client.py` |
| **입력 가드레일** | — (Cookbook 신규) | 주제 범위 체크 + 부적절 입력 필터 | **비동기 병렬 실행** (§52) | `guardrails/input.py` |
| **출력 검증** | — | Pydantic 스키마 + 할루시네이션 체크 | **Structured Outputs** (§54), **Pydantic** (§55~59) | `guardrails/output.py` |
| **사용자 피드백** | **YouTube Implicit Feedback(§48)** | 루틴 완료율, 건너뛴 운동, 선호 운동 | — | `feedback/` |
| **평가** | — (Cookbook 신규) | Retrieval Recall + Faithfulness + Safety | **Eval-driven 개발** (§53) | `eval/` |
| **디버깅** | — (Cookbook 신규) | 검색 실패 vs 생성 실패 분류 | **print_context 패턴** (§61) | 개발 도구 |

---

## 64. 기술적 의사결정을 위한 체크리스트 (최종)

### 문서 처리 (Cookbook §50 보강)
- [ ] **Chunking 전략**: 구조적 분할(헤더 기준) + 토큰 기반(max 1000 토큰) + 재귀적 분할
- [ ] **제목 보존**: 각 청크에 부모 제목(H1 > H2) prepend
- [ ] **노이즈 제거**: 참고문헌, 링크 등 불필요 섹션 제외

### Embedding 모델
- [x] **Embedding 모델 선택: `BAAI/bge-m3`** — 한국어 지원, 3가지 검색 통합, MIT 라이선스
- [ ] fp16 사용 여부: 속도 향상 vs 미세한 정확도 차이 확인
- [ ] max_length 설정: 문서 특성에 따라 512~8192 조정

### 검색 파이프라인
- [ ] Hybrid search 가중치 (w_dense, w_sparse): 초기 0.7/0.3, 실험 후 조정
- [ ] ColBERT 사용 여부: 정확도 중요 시 추가, 저장 비용 고려
- [ ] Top-K 값: 논문은 5~10, 도메인에 맞게 실험
- [ ] HyDE 적용 여부: Retrieval recall이 낮을 경우 도입 검토

### Re-ranker
- [x] **Reranker 선택: `bge-reranker-v2-m3`** — 다국어, 경량, BGE-M3와 동일 패밀리
- [ ] Reranker 입력 수: Top-N에서 N값 결정 (10~20)

### FAISS 인덱스
- [ ] **FAISS 인덱스 타입**: 초기 `IndexFlatIP(1024)` → 확장 시 `IndexIVFPQ` 전환 계획
- [ ] **nprobe 값**: 초기 10, recall 측정 후 조정
- [ ] **PQ 파라미터 (b, nbits)**: 규모 확장 시 b=8~16, nbits=8
- [ ] **인덱스 저장 경로**: `vectorstore/` 디렉토리

### 프롬프트 구성 (Cookbook §51, §60 + YouTube §48 보강)
- [ ] **Token budget**: max_model_len - 시스템프롬프트 - 예상응답길이 → 문서 삽입 동적 결정
- [ ] **"답변 불가" 처리**: 검색 문서에 답이 없을 때 할루시네이션 대신 에러 JSON 반환
- [ ] **프롬프트 구조**: 시스템 프롬프트 → 참고 문서 → 출력 형식 → 사용자 질문 순서

### LLM 서빙 (vLLM + Qwen2.5)
- [x] **서빙 엔진: vLLM** — PagedAttention으로 2~4배 처리량, OpenAI API 호환
- [x] **LLM 모델: `Qwen/Qwen2.5-7B-Instruct`** — 한국어 29+언어, 128K 컨텍스트, Apache 2.0, GQA
- [ ] GPU 사양: 7.6B 모델 → 단일 A100 40GB로 서빙 가능
- [ ] max_model_len: 4096~8192 (프롬프트 + 생성 토큰 합계)
- [ ] Shared prefix: 시스템 프롬프트 캐싱 활성화 여부
- [ ] Sampling 파라미터: temperature (0.3~0.5), top_p (0.8), repetition_penalty (1.05), max_tokens (512~1024)

### Structured Outputs & Pydantic 검증 (§54~§59 보강)
- [x] **검증 라이브러리: Pydantic v2** — Rust 기반 고성능, LLM 생태계 표준
- [ ] **Pydantic 모델 정의**: `Routine`, `Exercise` 모델 (타입 + 범위 제약 + enum)
- [ ] **JSON Schema 생성**: `model_json_schema()` → 프롬프트에 포함 또는 guided_json 전달
- [ ] **도메인 커스텀 검증**: `@field_validator`(reps 형식), `@model_validator`(시간 일관성, 위험 운동)
- [ ] **재시도 전략**: ValidationError의 `e.errors()` → 에러 피드백 프롬프트 → 최대 3회
- [ ] **Partial JSON**: `from_json(allow_partial=True)` — max_tokens 잘림 대응
- [ ] **guided decoding**: 정확도 문제 시 vLLM의 outlines 백엔드 도입 검토

### Guardrails (Cookbook §52 보강 — 신규)
- [ ] **입력 가드레일**: 주제 범위 체크 (피트니스 관련 질문만 허용)
- [ ] **출력 할루시네이션 체크**: 검색 문서 vs LLM 응답 교차 검증
- [ ] **위험 운동 필터**: 초보자에게 부적절한 고난이도 운동 경고
- [ ] **비동기 실행**: 가드레일과 LLM 호출 병렬 실행으로 레이턴시 최소화

### 추천 시스템 설계 (YouTube 논문 §43~§49 보강 — 신규)
- [ ] **2단계 아키텍처**: RAG 검색(=Candidate Generation) → LLM 생성(=Ranking) 매핑 확인
- [ ] **사용자 프로필 피처**: 경력, 목표, 가용 시간, 장비, 부상 이력 → 프롬프트에 반영
- [ ] **Implicit Feedback 수집**: 루틴 완료율, 건너뛴 운동, 선호 운동 기록
- [ ] **시간 편향 보정**: 운동 문서의 업데이트 시점 기반 time_decay_boost 적용
- [ ] **사용자 균등 가중**: 소수 활발 사용자가 피드백을 지배하지 않도록

### 평가 (Cookbook §53 보강 — 신규)
- [ ] **평가 세트 구축**: Q&A 쌍 + 기대 검색 문서 + 기대 응답 요소
- [ ] **Retrieval 평가**: Recall@K, MRR 측정
- [ ] **Generation 평가**: Faithfulness, Schema compliance
- [ ] **최종 평가**: 루틴 완료율 (YouTube의 Watch Time에 해당) — Offline ≠ Online 유의
- [ ] **연속적 평가**: 프로덕션 로그 → eval set 확장
