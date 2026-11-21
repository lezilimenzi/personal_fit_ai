"""JSON 문서를 검색에 최적화된 청크로 분할. RESEARCH.md §50 5단계 전략."""

import re


class Chunker:
    def __init__(self, tokenizer, max_tokens: int = 1000):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def chunk_all(self, docs: list[dict]) -> list[dict]:
        chunks = []
        for doc in docs:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        return chunks

    def _chunk_document(self, doc: dict) -> list[dict]:
        text = doc["text"]
        title = doc.get("title", "")

        sections = self._split_by_headers(text, title)

        chunks = []
        for i, section in enumerate(sections):
            section_title = section["title"]
            section_text = section["text"]

            # 노이즈 제거
            section_text = self._remove_noise(section_text)
            if not section_text.strip():
                continue

            full_text = f"{section_title}\n\n{section_text}" if section_title else section_text
            token_count = self._count_tokens(full_text)

            if token_count <= self.max_tokens:
                chunks.append(self._make_chunk(doc, i, section_title, full_text, token_count))
            else:
                sub_chunks = self._recursive_split(full_text, section_title)
                for j, sc in enumerate(sub_chunks):
                    tc = self._count_tokens(sc)
                    chunks.append(self._make_chunk(doc, i * 100 + j, section_title, sc, tc))

        if not chunks:
            tc = self._count_tokens(text)
            chunks.append(self._make_chunk(doc, 0, title, text, tc))

        return chunks

    def _split_by_headers(self, text: str, parent_title: str) -> list[dict]:
        pattern = r"^(#{2,3})\s+(.+)$"
        parts = re.split(pattern, text, flags=re.MULTILINE)

        if len(parts) == 1:
            return [{"title": parent_title, "text": text}]

        sections = []
        if parts[0].strip():
            sections.append({"title": parent_title, "text": parts[0].strip()})

        i = 1
        while i < len(parts):
            header_text = parts[i + 1] if i + 1 < len(parts) else ""
            body = parts[i + 2].strip() if i + 2 < len(parts) else ""
            section_title = f"{parent_title} > {header_text}" if parent_title else header_text
            sections.append({"title": section_title, "text": body})
            i += 3

        return sections

    def _recursive_split(self, text: str, title: str) -> list[str]:
        """재귀적 분할: \n\n → \n → . 순서로 경계 탐색."""
        for separator in ["\n\n", "\n", ". "]:
            parts = text.split(separator)
            if len(parts) <= 1:
                continue

            chunks = []
            current = ""
            for part in parts:
                candidate = current + separator + part if current else part
                if self._count_tokens(candidate) <= self.max_tokens:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part
            if current:
                chunks.append(current)

            if all(self._count_tokens(c) <= self.max_tokens for c in chunks):
                return chunks

        # 최후 수단: 토큰 단위 강제 분할
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

    def _remove_noise(self, text: str) -> str:
        text = re.sub(r"^참고문헌.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r"^references.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r"https?://\S+", "", text)
        return text.strip()

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _make_chunk(self, doc: dict, idx: int, title: str, text: str,
                    token_count: int) -> dict:
        return {
            "chunk_id": f"{doc['doc_id']}_{idx:03d}",
            "doc_id": doc["doc_id"],
            "title": title,
            "text": text,
            "category": doc.get("category", ""),
            "updated_at": doc.get("updated_at", ""),
            "token_count": token_count,
        }
