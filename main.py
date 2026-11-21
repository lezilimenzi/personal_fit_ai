"""Personal Fit AI — CLI 엔트리포인트."""

import asyncio
import json
import sys

import config
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import HybridRetriever
from src.rag.reranker import Reranker
from src.rag.prompt_builder import PromptBuilder
from src.llm.vllm_client import VLLMClient
from src.validation.routine_schema import Routine
from src.validation.routine_validator import RoutineValidator
from src.guardrails.input_guard import InputGuard
from src.guardrails.output_guard import OutputGuard
from src.app.recommend_service import RecommendService

from transformers import AutoTokenizer


def collect_signup_profile() -> dict:
    """가입 시 1회 입력: 나이, 성별, BMI, 골격근량(선택), 운동경력, 목표, 부상이력."""
    print("\n=== 사용자 프로필 입력 (가입 시 1회) ===")
    age = input("나이: ").strip()
    gender = input("성별 (남/여): ").strip()
    bmi = input("BMI: ").strip()
    skeletal = input("골격근량 (선택, 없으면 Enter): ").strip()
    experience = input("운동경력 (초보/중급/고급): ").strip()
    goal = input("목표 (근력 증가/체지방 감소/체력 향상/유연성): ").strip()
    injuries = input("부상이력 (없으면 Enter): ").strip()

    return {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "skeletal_muscle_mass": skeletal or None,
        "experience": experience,
        "goal": goal,
        "injuries": injuries or "없음",
    }


def collect_daily_input() -> dict:
    """매 접속 시 입력: 가용시간, 장비, 어제 실시한 운동."""
    print("\n=== 오늘의 조건 입력 (매 접속 시) ===")
    time_min = input("가용 시간 (분): ").strip()
    equipment = input("장비 (예: 바벨, 덤벨, 맨몸 등): ").strip()
    yesterday = input("어제 실시한 운동 (없으면 Enter): ").strip()

    return {
        "time": time_min,
        "equipment": equipment,
        "yesterday_workout": yesterday or "없음",
    }


def format_routine(routine: Routine) -> str:
    lines = [
        f"\n{'='*50}",
        f"  {routine.routine_name}",
        f"  난이도: {routine.difficulty.value} | 타겟: {routine.target_muscle}",
        f"  총 소요시간: {routine.total_time_minutes}분",
        f"{'='*50}",
    ]
    for i, ex in enumerate(routine.exercises, 1):
        lines.append(f"\n  {i}. {ex.name}")
        lines.append(f"     {ex.sets}세트 × {ex.reps}회 | 휴식 {ex.rest_seconds}초")
        if ex.notes:
            lines.append(f"     💡 {ex.notes}")

    if routine.warnings:
        lines.append(f"\n{'─'*50}")
        lines.append("  ⚠ 주의사항:")
        for w in routine.warnings:
            lines.append(f"    - {w}")

    return "\n".join(lines)


def main():
    # 1. LLM 서버 연결 확인
    llm_client = VLLMClient(
        base_url=config.VLLM_BASE_URL,
        model=config.LLM_MODEL,
        api_key=config.VLLM_API_KEY,
    )
    print("vLLM 서버 연결 확인 중...")
    if not llm_client.health_check():
        print("❌ vLLM 서버에 연결할 수 없습니다.")
        print(f"   VLLM_BASE_URL: {config.VLLM_BASE_URL}")
        print("   Colab 노트북이 실행 중인지 확인하세요.")
        sys.exit(1)
    print("✅ vLLM 서버 연결 성공\n")

    # 2. 컴포넌트 초기화
    print("모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL, trust_remote_code=True)
    embedder = Embedder(config.EMBEDDING_MODEL)

    store = VectorStore()
    store.load(config.VECTORSTORE_DIR)
    print(f"  벡터 인덱스 로드: {store.index.ntotal}개 청크")

    retriever = HybridRetriever(
        embedder, store,
        w_dense=config.W_DENSE,
        w_sparse=config.W_SPARSE,
    )
    reranker = Reranker(config.RERANKER_MODEL)

    json_schema = json.dumps(Routine.model_json_schema(), ensure_ascii=False, indent=2)
    prompt_builder = PromptBuilder(
        tokenizer=tokenizer,
        max_model_len=config.MAX_MODEL_LEN,
        max_tokens=config.MAX_TOKENS,
        json_schema=json_schema,
    )

    validator = RoutineValidator(llm_client, max_retries=config.MAX_VALIDATION_RETRIES)
    input_guard = InputGuard()
    output_guard = OutputGuard()

    service = RecommendService(
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
        validator=validator,
        input_guard=input_guard,
        output_guard=output_guard,
    )

    # 3. 사용자 프로필 (가입 시 1회)
    profile = collect_signup_profile()

    # 4. 추천 루프
    print("\n✅ 준비 완료! 운동 루틴을 추천받으세요. (종료: quit)")

    while True:
        daily = collect_daily_input()
        profile["daily"] = daily

        query = input("\n운동 루틴 요청 (예: '가슴 운동 루틴 추천해줘'): ").strip()
        if query.lower() in ("quit", "exit", "종료"):
            print("종료합니다.")
            break

        if not query:
            query = f"{profile.get('goal', '운동')} 루틴 추천해주세요"

        try:
            print("\n추천 생성 중...")
            routine = asyncio.run(service.recommend(query, profile))
            print(format_routine(routine))
        except ValueError as e:
            print(f"\n⚠ {e}")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
