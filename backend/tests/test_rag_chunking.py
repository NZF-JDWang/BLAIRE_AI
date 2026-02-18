from pathlib import Path

from app.rag.chunking import chunk_text, extract_text


def test_chunk_text_generates_multiple_chunks() -> None:
    text = "a" * 2200
    chunks = chunk_text(text, chunk_size=900, overlap=150)
    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0


def test_extract_text_markdown(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.md"
    file_path.write_text("# Title", encoding="utf-8")
    text = extract_text(file_path)
    assert "Title" in text


def test_extract_text_pdf_returns_non_empty(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-1.4\n% fake")
    text = extract_text(file_path)
    assert text.strip() != ""


def test_extract_text_image_returns_non_empty(tmp_path: Path) -> None:
    file_path = tmp_path / "photo.png"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    text = extract_text(file_path)
    assert text.strip() != ""
