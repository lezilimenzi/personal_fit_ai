### 🚀 핵심 워크플로우 4단계
**대전제: 작성된 계획을 사용자가 직접 검토하고 승인하기전까지는 절대로 코드를 작성하지 않는다.**
기획과 코딩의 분리!!!

1.  리서치
`research.md` 파일을 만들어 관련 코드를 깊이 이해하게 합니다.
AI가 대충 훑어보지 않도록 '깊이', '매우 상세히' 같은 표현을 프롬프트에 사용해야 합니다.
2.  상세 구현 계획 수립
`plan.md` 파일에 구현 방식, 코드 스니펫, 변경될 파일 경로, 트레이드 오프를 포함한 계획을 작성하게 합니다.
3.  반복적 계획 검토 및 수정
작성된 `plan.md`를 에디터에서 직접 열어 메모(주석)를 추가하여 잘못된 가정을 수정하거나 제약 조건을 추가합니다.
이 과정을 AI가 납득할 때까지 반복하며, "아직 구현하지 마"라고 명시해야 합니다.
4.  자동화된 구현
계획이 확정되면 구현 명령을 내립니다. 이때 코딩은 기계적으로 이루어지며, 사용자는 '감독자' 역할을 합니다.

### 💡 핵심 팁
내장 플랫 모드 대신 `plan.md` 활용: 에디터에서 직접 편집 가능하고 프로젝트 산출물로 남겨 연속성을 유지합니다 (4:05).
잘못된 방향은 `git reset`: 잘못된 접근을 조금씩 고치려 하지 말고, 과감히 되돌리고 범위를 재설정하는 것이 더 효율적입니다.
하나의 긴 세션 사용: 리서치부터 구현까지 하나의 세션에서 진행하여 컨텍스트를 유지합니다.

---
name: karpathy-guidelines
description: Behavioral guidelines to reduce common LLM coding mistakes. Use when writing, reviewing, or refactoring code to avoid overcomplication, make surgical changes, surface assumptions, and define verifiable success criteria.
license: MIT
---

# Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.