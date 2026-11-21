"""원본 파일(md, PDF, CSV, HTML, JSON)을 통일된 JSON 문서 리스트로 변환."""

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def _today() -> str:
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# 개별 파일 로더
# ---------------------------------------------------------------------------

def load_markdown(filepath: str) -> list[dict]:
    text = Path(filepath).read_text(encoding="utf-8")
    stem = Path(filepath).stem
    return [{
        "doc_id": stem,
        "title": stem.replace("_", " ").title(),
        "source": filepath,
        "category": _infer_category(filepath),
        "updated_at": _today(),
        "text": text,
    }]


def load_pdf(filepath: str) -> list[dict]:
    import fitz  # pymupdf
    doc = fitz.open(filepath)
    pages = [page.get_text() for page in doc]
    text = "\n".join(pages)
    doc.close()
    stem = Path(filepath).stem
    return [{
        "doc_id": stem,
        "title": stem.replace("_", " ").title(),
        "source": filepath,
        "category": _infer_category(filepath),
        "updated_at": _today(),
        "text": text,
    }]


def load_html(filepath: str) -> list[dict]:
    raw = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    stem = Path(filepath).stem
    return [{
        "doc_id": stem.lower().replace(" ", "_"),
        "title": stem,
        "source": filepath,
        "category": "guideline",
        "updated_at": _today(),
        "text": text,
    }]


def _infer_category(filepath: str) -> str:
    fp = filepath.lower()
    if "safety" in fp:
        return "safety"
    if "program" in fp:
        return "program"
    if "exercise" in fp:
        return "exercise_guide"
    return "knowledge"


# ---------------------------------------------------------------------------
# recommend_data/ 전용 변환
# ---------------------------------------------------------------------------

def load_gym_csv(filepath: str) -> list[dict]:
    """GYM.csv → Goal × BMI Category 그룹핑 → ~16개 RAG 문서."""
    df = pd.read_csv(filepath)
    docs = []
    grouped = df.groupby(["Goal", "BMI Category"]).first().reset_index()

    for _, row in grouped.iterrows():
        goal = row["Goal"]
        bmi_cat = row["BMI Category"]
        doc_id = f"gym_{goal}_{bmi_cat.lower().replace(' ', '_')}"
        title = f"{goal} 목표 — {bmi_cat} 권장 루틴"
        text = (
            f"목표: {goal}\nBMI 분류: {bmi_cat}\n\n"
            f"권장 운동: {row['Exercise Schedule']}\n\n"
            f"권장 식단: {row['Meal Plan']}"
        )
        docs.append({
            "doc_id": doc_id,
            "title": title,
            "source": filepath,
            "category": "gym_routine",
            "updated_at": _today(),
            "text": text,
        })
    return docs


def load_exercise_csv(filepath: str) -> list[dict]:
    """exercise.csv → 각 행을 자연어 운동 설명 문서로 변환."""
    df = pd.read_csv(filepath)
    docs = []
    for _, row in df.iterrows():
        doc_id = row["exercise_id"]
        text = (
            f"운동: {row['name_ko']}\n"
            f"- 패턴: {row['pattern']}\n"
            f"- 주요 근육: {row['muscle_group']}\n"
            f"- 필요 장비: {row['equipment_need']}"
        )
        docs.append({
            "doc_id": doc_id,
            "title": row["name_ko"],
            "source": filepath,
            "category": "exercise_catalog",
            "updated_at": _today(),
            "text": text,
        })
    return docs


def load_free_exercise_db(filepath: str) -> list[dict]:
    """free_exercise_db.json → 873개 운동 수행 가이드 문서."""
    with open(filepath, "r", encoding="utf-8") as f:
        exercises = json.load(f)

    docs = []
    for ex in exercises:
        instructions = "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(ex.get("instructions", []))
        )
        primary = ", ".join(ex.get("primaryMuscles", []))
        secondary = ", ".join(ex.get("secondaryMuscles", []))
        text = (
            f"Exercise: {ex['name']}\n"
            f"Level: {ex.get('level', 'N/A')}\n"
            f"Equipment: {ex.get('equipment', 'N/A')}\n"
            f"Category: {ex.get('category', 'N/A')}\n"
            f"Primary Muscles: {primary}\n"
            f"Secondary Muscles: {secondary or 'None'}\n\n"
            f"Instructions:\n{instructions}"
        )
        docs.append({
            "doc_id": ex.get("id", ex["name"].lower().replace(" ", "_")),
            "title": ex["name"],
            "source": filepath,
            "category": "exercise_detail",
            "updated_at": _today(),
            "text": text,
        })
    return docs


# ---------------------------------------------------------------------------
# 메인 로더
# ---------------------------------------------------------------------------

class DocumentLoader:
    def __init__(self, raw_knowledge_dir: str, raw_catalog_dir: str,
                 recommend_data_dir: str):
        self.raw_knowledge_dir = raw_knowledge_dir
        self.raw_catalog_dir = raw_catalog_dir
        self.recommend_data_dir = recommend_data_dir

    def load_all(self) -> list[dict]:
        docs = []

        # 1. data/raw/knowledge/ (md, PDF)
        knowledge_dir = Path(self.raw_knowledge_dir)
        if knowledge_dir.exists():
            for fp in knowledge_dir.rglob("*"):
                if fp.suffix == ".md":
                    docs.extend(load_markdown(str(fp)))
                elif fp.suffix == ".pdf":
                    docs.extend(load_pdf(str(fp)))

        # 2. data/raw/catalog/exercise.csv
        catalog_csv = Path(self.raw_catalog_dir) / "exercise.csv"
        if catalog_csv.exists():
            docs.extend(load_exercise_csv(str(catalog_csv)))

        # 3. recommend_data/
        rec_dir = Path(self.recommend_data_dir)
        if rec_dir.exists():
            gym_csv = rec_dir / "GYM.csv"
            if gym_csv.exists():
                docs.extend(load_gym_csv(str(gym_csv)))

            exercise_csv = rec_dir / "exercise.csv"
            if exercise_csv.exists():
                docs.extend(load_exercise_csv(str(exercise_csv)))

            free_db = rec_dir / "free_exercise_db.json"
            if free_db.exists():
                docs.extend(load_free_exercise_db(str(free_db)))

            for html_file in rec_dir.glob("*.html"):
                docs.extend(load_html(str(html_file)))

            for pdf_file in rec_dir.glob("*.pdf"):
                docs.extend(load_pdf(str(pdf_file)))

        return docs

    def save_docs(self, docs: list[dict], output_dir: str) -> None:
        """카테고리별로 JSON 파일 저장."""
        os.makedirs(output_dir, exist_ok=True)
        by_category: dict[str, list] = {}
        for doc in docs:
            cat = doc["category"]
            by_category.setdefault(cat, []).append(doc)

        category_to_filename = {
            "gym_routine": "gym_routine_docs.json",
            "exercise_catalog": "exercise_docs.json",
            "exercise_detail": "exercise_detail_docs.json",
            "guideline": "guideline_docs.json",
            "exercise_guide": "exercise_docs.json",
            "program": "program_docs.json",
            "safety": "safety_docs.json",
            "knowledge": "knowledge_docs.json",
        }

        merged: dict[str, list] = {}
        for cat, cat_docs in by_category.items():
            filename = category_to_filename.get(cat, f"{cat}_docs.json")
            merged.setdefault(filename, []).extend(cat_docs)

        for filename, file_docs in merged.items():
            out_path = Path(output_dir) / filename
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(file_docs, f, ensure_ascii=False, indent=2)
