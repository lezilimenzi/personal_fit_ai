"""LLM 출력의 JSON 구조를 정의하는 Pydantic 모델."""

import re
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


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
        # '8-12', '10', '8-12 per leg', '10 each' 등 허용
        if not re.match(r"^\d+(-\d+)?(\s+[a-zA-Z\s]+)?$", v.strip()):
            raise ValueError(f"reps는 '8-12' 또는 '10' 형식이어야 합니다: {v}")
        return v.strip()


class Routine(BaseModel):
    routine_name: str
    difficulty: Difficulty
    target_muscle: str
    exercises: List[Exercise] = Field(min_length=1, max_length=10)
    total_time_minutes: int = Field(ge=10, le=120)
    warnings: List[str] = []

    @model_validator(mode="after")
    def check_time_consistency(self):
        min_time = len(self.exercises) * 3
        if self.total_time_minutes < min_time:
            raise ValueError(
                f"운동 {len(self.exercises)}개에 총 {self.total_time_minutes}분은 비현실적입니다"
            )
        return self
