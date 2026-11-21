"""전체 RAG 파이프라인을 하나의 recommend() 메서드로 오케스트레이션."""

import asyncio

import config
from src.validation.routine_schema import Routine


class RecommendService:
    def __init__(self, retriever, reranker, prompt_builder,
                 llm_client, validator, input_guard, output_guard):
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.validator = validator
        self.input_guard = input_guard
        self.output_guard = output_guard

    async def recommend(self, query: str,
                        user_profile: dict | None = None) -> Routine:
        # 1. 입력 가드레일
        guard_ok, reason = await self.input_guard.check(query)
        if not guard_ok:
            raise ValueError(reason)

        # 2. 검색
        candidates = self.retriever.retrieve(query, top_n=config.RETRIEVAL_TOP_N)

        # 3. 리랭킹
        top_docs = self.reranker.rerank(query, candidates, top_k=config.RERANK_TOP_K)

        # 4. 프롬프트 구성
        messages = self.prompt_builder.build(query, top_docs, user_profile)

        # 5. LLM 생성
        raw_output = self.llm_client.generate(
            messages,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_TOKENS,
            repetition_penalty=config.REPETITION_PENALTY,
        )

        # 6. 검증 + 재시도
        routine = self.validator.validate_and_retry(raw_output, messages)

        # 7. 출력 가드레일
        _, warnings = self.output_guard.check(routine, top_docs)
        if warnings:
            routine.warnings.extend(warnings)

        return routine
