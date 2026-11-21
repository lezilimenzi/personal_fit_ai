"""LLM 출력 → Pydantic 검증 → 실패 시 에러 피드백 포함하여 재시도."""

import json
import re

from pydantic import ValidationError

import config
from src.validation.routine_schema import Routine


def _extract_json(text: str) -> str:
    """LLM 응답에서 JSON 블록을 추출 (```json ... ``` 또는 raw JSON)."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


class RoutineValidator:
    def __init__(self, llm_client, max_retries: int = 3):
        self.llm_client = llm_client
        self.max_retries = max_retries

    def validate_and_retry(self, raw_output: str,
                           original_messages: list[dict]) -> Routine:
        for attempt in range(self.max_retries):
            try:
                json_str = _extract_json(raw_output)
                return Routine.model_validate_json(json_str)
            except (ValidationError, json.JSONDecodeError) as e:
                if attempt == self.max_retries - 1:
                    raise
                error_feedback = self._format_errors(e)
                retry_messages = original_messages + [
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            f"JSON 검증 실패:\n{error_feedback}\n"
                            "위 오류를 수정하여 다시 JSON을 생성하세요."
                        ),
                    },
                ]
                raw_output = self.llm_client.generate(
                    retry_messages,
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    repetition_penalty=config.REPETITION_PENALTY,
                )

    def _format_errors(self, e: Exception) -> str:
        if isinstance(e, ValidationError):
            lines = []
            for err in e.errors():
                loc = " → ".join(str(part) for part in err["loc"])
                lines.append(f"- {loc}: {err['msg']}")
            return "\n".join(lines)
        return str(e)
