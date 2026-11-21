# Personal Fit AI

> 개인 학습·포트폴리오용으로 진행한 프로젝트다.

RAG와 LLM으로 개인 맞춤 운동 루틴을 추천하는 시스템이다. 사용자 프로필(나이, 성별, BMI, 골격근량, 운동경력, 목표, 부상이력)과 그날의 조건(가용 시간, 장비, 어제 한 운동)을 받아 운동 지식 문서를 검색하고 그 결과를 근거로 JSON 형식의 루틴을 만든다.

## 파이프라인

1. **입력 가드레일**: 피트니스와 관련된 질문인지 확인하고 프롬프트 인젝션을 거른다.
2. **임베딩**: `BAAI/bge-m3`로 질의의 dense 벡터(1024차원)와 sparse(lexical) 가중치를 함께 구한다.
3. **하이브리드 검색**: FAISS dense 검색과 sparse 매칭 결과를 `0.7 × dense + 0.3 × sparse`로 합친다.
4. **리랭킹**: `bge-reranker-v2-m3` cross-encoder로 상위 문서를 다시 고른다.
5. **프롬프트 구성**: 시스템 프롬프트, 사용자 프로필, 검색 문서, JSON 출력 스키마를 토큰 예산 안에서 조합한다.
6. **생성**: Google Colab T4에서 vLLM으로 띄운 `Qwen/Qwen2.5-7B-Instruct-AWQ`를 ngrok 터널로 호출한다.
7. **출력 검증**: Pydantic v2로 스키마와 도메인 규칙을 검사하고 실패하면 에러 내용을 넣어 최대 3회 다시 생성한다.
8. **출력 가드레일**: 검색 문서에 없는 운동(할루시네이션)과 초보자에게 위험한 운동을 걸러낸다.

## 디렉토리 구조

```
config.py                    # 모델·검색·생성 파라미터와 경로
main.py                      # CLI 엔트리포인트
scripts/build_vector_index.py  # 문서 로드 → 청킹 → 임베딩 → FAISS 인덱스 저장
notebooks/colab_llm_server.ipynb  # Colab vLLM 서버 + ngrok 터널
src/
  rag/          # document_loader, chunker, embedder, vector_store, retriever, reranker, prompt_builder
  llm/          # vLLM(OpenAI 호환) 클라이언트
  validation/   # 루틴 Pydantic 스키마와 검증·재시도
  guardrails/   # 입력/출력 가드레일
  app/          # 파이프라인 오케스트레이션
RESEARCH.md     # 기술 조사
PLAN.md         # 상세 구현 계획
ERROR.md        # 개발 중 겪은 오류와 해결 기록
```

## 데이터

저작권과 용량 문제로 원본 데이터, 논문 PDF, 가공 산출물은 저장소에 넣지 않았다. 직접 실행하려면 아래 파일을 준비해야 한다. 각 데이터의 내용과 쓰임새는 [PLAN.md](PLAN.md) 7장에 정리해 두었다.

- `recommend_data/`: `GYM.csv`, `exercise.csv`, `free_exercise_db.json`, 운동 가이드라인 HTML, 운동과학 논문 PDF, Fitbit 활동 데이터(2단계용)
- `data/raw/catalog/exercise.csv`, `data/raw/knowledge/{exercises,programs,safety}/`: 선택 사항인 운동 카탈로그와 지식 문서(Markdown)
- `data/docs/`, `vectorstore/`: 아래 인덱스 빌드 단계에서 생성된다.

## 실행

Python 3.10 가상환경에서 개발했다.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 1. 벡터 인덱스 빌드 (recommend_data/ 준비 후)
python scripts/build_vector_index.py

# 2. Colab에서 notebooks/colab_llm_server.ipynb 실행 → 출력된 ngrok 주소를 지정
$env:VLLM_BASE_URL="https://<ngrok-주소>/v1"

# 3. CLI 실행
python main.py
```
