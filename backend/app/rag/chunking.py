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
        return _extract_with_llamaindex_or_fallback(file_path, fallback=f"[PDF content] file={file_path.name}")
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return _extract_with_llamaindex_or_fallback(file_path, fallback=f"[Image content] file={file_path.name}")
    return ""


def _extract_with_llamaindex_or_fallback(file_path: Path, fallback: str) -> str:
    try:
        from llama_index.core import SimpleDirectoryReader  # type: ignore[import-untyped]

        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        docs = reader.load_data()
        text_parts = [str(getattr(doc, "text", "")).strip() for doc in docs]
        merged = "\n".join([part for part in text_parts if part])
        return merged or fallback
    except Exception:  # noqa: BLE001
        return fallback


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
