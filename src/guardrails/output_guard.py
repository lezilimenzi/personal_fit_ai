"""출력 가드레일: 할루시네이션 체크, 위험 운동 필터."""

from src.validation.routine_schema import Routine

DANGEROUS_EXERCISES_FOR_BEGINNERS = {
    "behind the neck press", "behind neck press",
    "upright row", "업라이트 로우",
    "good morning", "굿모닝",
    "barbell hack squat",
}


class OutputGuard:
    def check(self, routine: Routine,
              retrieved_docs: list[dict]) -> tuple[bool, list[str]]:
        warnings = []

        # 1. 할루시네이션 체크: 운동명이 검색 문서에 존재하는지
        doc_texts = " ".join(doc.get("text", "") for doc in retrieved_docs).lower()
        for ex in routine.exercises:
            name_lower = ex.name.lower()
            if name_lower not in doc_texts:
                warnings.append(
                    f"'{ex.name}'은(는) 참고 문서에서 확인되지 않은 운동입니다."
                )

        # 2. 위험 운동 필터: 초보자에게 고위험 운동 추천 시 경고
        if routine.difficulty == "beginner":
            for ex in routine.exercises:
                name_lower = ex.name.lower()
                for dangerous in DANGEROUS_EXERCISES_FOR_BEGINNERS:
                    if dangerous in name_lower:
                        warnings.append(
                            f"'{ex.name}'은(는) 초보자에게 부상 위험이 높은 운동입니다. "
                            "대체 운동을 고려하세요."
                        )

        safe = len(warnings) == 0
        return safe, warnings
