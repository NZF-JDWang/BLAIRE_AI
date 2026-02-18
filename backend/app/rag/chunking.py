from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return f"[PDF content placeholder] file={file_path.name}"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return f"[Image content placeholder] file={file_path.name}"
    return ""


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[TextChunk]:
    if not text.strip():
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    normalized = " ".join(text.split())
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(TextChunk(chunk_index=index, text=piece))
            index += 1
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks

