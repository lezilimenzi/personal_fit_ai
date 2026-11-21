"""검색 문서 + 사용자 정보 + JSON 스키마를 token budget 내에서 조립."""

SYSTEM_PROMPT_TEMPLATE = """당신은 전문 피트니스 코치입니다.
아래 참고 문서만을 기반으로 운동 루틴을 추천합니다.
참고 문서에 관련 정보가 없으면 "정보가 부족합니다"라고 답하세요.

{user_profile_section}

반드시 아래 JSON 형식으로만 응답하세요:
{json_schema}"""


def _build_user_profile_section(profile: dict | None) -> str:
    if not profile:
        return ""

    lines = ["[사용자 프로필]  ← 가입 시 1회 입력"]
    field_map = [
        ("age", "나이"),
        ("gender", "성별"),
        ("bmi", "BMI"),
        ("experience", "운동경력"),
        ("goal", "목표"),
        ("injuries", "부상이력"),
    ]
    for key, label in field_map:
        if key in profile and profile[key]:
            lines.append(f"- {label}: {profile[key]}")

    if profile.get("skeletal_muscle_mass"):
        lines.append(f"- 골격근량: {profile['skeletal_muscle_mass']}")

    daily = profile.get("daily", {})
    if daily:
        lines.append("\n[오늘의 조건]  ← 매 접속 시 입력")
        if daily.get("time"):
            lines.append(f"- 가용 시간: {daily['time']}분")
        if daily.get("equipment"):
            lines.append(f"- 장비: {daily['equipment']}")
        if daily.get("yesterday_workout"):
            lines.append(f"- 어제 실시한 운동: {daily['yesterday_workout']}")

    return "\n".join(lines)


class PromptBuilder:
    def __init__(self, tokenizer, max_model_len: int, max_tokens: int,
                 json_schema: str):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.json_schema = json_schema
        self.system_prompt_budget = 500
        self.doc_budget = max_model_len - max_tokens - self.system_prompt_budget

    def build(self, query: str, docs: list[dict],
              user_profile: dict | None = None) -> list[dict]:
        profile_section = _build_user_profile_section(user_profile)
        system = SYSTEM_PROMPT_TEMPLATE.format(
            user_profile_section=profile_section,
            json_schema=self.json_schema,
        )
        context = self._insert_docs_within_budget(docs)

        user_content = query
        if context:
            user_content = f"{context}\n\n질문: {query}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _insert_docs_within_budget(self, docs: list[dict]) -> str:
        result = ""
        remaining = self.doc_budget
        for doc in docs:
            doc_text = f"\n\n[참고 문서: {doc.get('title', '')}]\n{doc['text']}"
            tokens = len(self.tokenizer.encode(doc_text))
            if tokens > remaining:
                break
            result += doc_text
            remaining -= tokens
        return result
