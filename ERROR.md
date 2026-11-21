# ERROR.md — 런타임 에러 원인 및 해결방법

> 프로젝트 실행/운영 중 발생한 에러와 해결 방법을 기록합니다.
> 앞으로 모든 수정 사항은 여기에 **원인 → 해결방법** 순으로 정리합니다.

---

## 1. ngrok: vLLM 서버에 연결할 수 없음

### 증상
- `main.py` 실행 시 "❌ vLLM 서버에 연결할 수 없습니다" 출력
- Colab + ngrok URL 사용 중, `lsof`로는 vLLM이 8000 포트에서 리스닝 중임을 확인

### 원인
- ngrok 무료 tier는 브라우저가 아닌 **API 요청**에도 "Visit Site" 경고 HTML 페이지를 먼저 반환함
- vLLM 클라이언트가 JSON 대신 HTML을 수신하여 연결 실패

### 해결방법
- `src/llm/vllm_client.py`에 ngrok URL 사용 시 `ngrok-skip-browser-warning` 헤더 추가:
  ```python
  if "ngrok" in base_url:
      headers["ngrok-skip-browser-warning"] = "true"
  ```

---

## 2. Colab: localhost:8000 connection refused (5분 대기 타임아웃)

### 증상
- `colab_llm_server.ipynb` 셀 2 실행 후 5분 대기해도 "서버가 아직 준비되지 않았을 수 있습니다" 출력
- 별도 셀에서 `!lsof -i:8000` 실행 시 vLLM이 포트 8000에서 리스닝 중으로 표시됨

### 원인
- Colab에서 `localhost`가 IPv6 주소 `::1`로 해석됨
- vLLM은 IPv4(`0.0.0.0:8000`)만 리스닝
- Python `requests.get("http://localhost:8000")`가 `::1`로 접속 시도 → connection refused

### 해결방법
- 대기 루프에서 `localhost` 대신 **`127.0.0.1`** 사용 (IPv4 명시):
  ```python
  r = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
  ```
- ngrok 터널 연결 시 `ngrok.connect("127.0.0.1:8000")` 사용 권장

---

## 3. vLLM 400 BadRequestError: context length 초과

### 증상
```
Error code: 400 - {'error': {'message': "You passed 3073 input tokens and requested 1024 output tokens. However, the model's context length is only 4096 tokens, resulting in a maximum input length of 3072 tokens. Please reduce the length of the input prompt."}}
```

### 원인
- 모델 컨텍스트: 4096 토큰, 출력 예약: 1024 토큰 → 입력 한도 3072 토큰
- 실제 입력(시스템 프롬프트 + 검색 문서 + 질의) = 3073 토큰 → 1토큰 초과

### 해결방법
- `config.py`에서 `MAX_TOKENS`를 **1024 → 512**로 축소 (입력 한도 3584 토큰)
- `RERANK_TOP_K`를 **5 → 4**로 축소 (검색 문서 수 감소, 특히 "전신" 등 광범위 질의 시)
- `src/validation/routine_validator.py`: 재시도 시 `max_tokens` 등 `generate` 파라미터 전달 (기본값 1024 사용 시 재시도 메시지가 더 길어져 토큰 초과 발생)
- `src/llm/vllm_client.py`: `max_tokens` 기본값을 1024 → 512로 변경

---

## 3-1. 토큰 에러 반복 (전신 운동 등)

### 증상
- "전신 운동 루틴" 같은 질의에서 토큰 에러가 반복 발생
- "requested 1024 output tokens" 메시지로 768 설정이 적용되지 않은 것처럼 보임

### 원인
- **재시도 경로**: Pydantic 검증 실패 시 `RoutineValidator`가 `llm_client.generate(retry_messages)` 호출 시 `max_tokens`를 넘기지 않음 → 기본값 1024 사용
- **전신 질의**: "전신"은 여러 부위(가슴, 등, 하체 등)에 매칭되어 검색 문서 수가 늘어남 → 입력 프롬프트가 길어짐
- 재시도 시 메시지에 에러 피드백이 추가되어 입력이 더 길어짐

### 해결방법
- `src/validation/routine_validator.py`: 재시도 시 `config.MAX_TOKENS`, `config.TEMPERATURE` 등 생성 파라미터 전달
- `config.py`: `MAX_TOKENS` **768 → 512**로 축소 (입력 여유 3584 토큰)
- `config.py`: `RERANK_TOP_K` **5 → 4**로 축소 (검색 문서 수 감소)
- `src/llm/vllm_client.py`: `max_tokens` 기본값 **1024 → 768**로 변경 (누락 시에도 안전)

---

## 4. Pydantic validation error: reps 형식

### 증상
```
Value error, reps는 '8-12' 또는 '10' 형식이어야 합니다: 8-12 per leg
```

### 원인
- LLM이 한쪽 다리/팔 운동에 대해 `"8-12 per leg"`, `"10 each"` 같은 형식으로 출력
- `routine_schema.py`의 `reps` 검증 정규식이 `^\d+(-\d+)?$`로만 허용 → `per leg` 등 접미사 거절

### 해결방법
- `src/validation/routine_schema.py`의 `validate_reps_format`을 완화:
  - `^\d+(-\d+)?(\s+[a-zA-Z\s]+)?$` 사용해 `"8-12 per leg"`, `"10 each"` 등 허용
