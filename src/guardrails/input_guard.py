"""입력 가드레일: 피트니스 무관 질문, 프롬프트 인젝션 차단.
1단계에서는 키워드 기반 간단 구현."""

FITNESS_KEYWORDS = {
    "운동", "루틴", "트레이닝", "헬스", "웨이트", "스쿼트", "벤치프레스", "데드리프트",
    "근력", "유산소", "스트레칭", "워밍업", "세트", "반복", "근육", "가슴", "등",
    "어깨", "하체", "이두", "삼두", "복근", "코어", "풀업", "푸시업", "런지",
    "exercise", "workout", "routine", "training", "fitness", "muscle", "set", "rep",
    "체중", "다이어트", "벌크업", "컷", "린매스", "프로그램", "분할",
}

INJECTION_PATTERNS = [
    "ignore previous",
    "ignore above",
    "disregard",
    "system prompt",
    "새로운 역할",
    "역할을 바꿔",
    "명령을 무시",
]


class InputGuard:
    async def check(self, user_input: str) -> tuple[bool, str]:
        lower = user_input.lower()

        for pattern in INJECTION_PATTERNS:
            if pattern in lower:
                return False, "부적절한 입력이 감지되었습니다."

        has_fitness_keyword = any(kw in user_input for kw in FITNESS_KEYWORDS)
        if not has_fitness_keyword and len(user_input) > 10:
            return False, "피트니스와 관련된 질문을 입력해주세요."

        return True, ""
