from pathlib import Path

import pytest

from app.services.filesystem_sandbox import FilesystemSandbox, FilesystemSandboxError


def test_validate_allows_path_within_root(tmp_path: Path) -> None:
    sandbox = FilesystemSandbox([str(tmp_path)])
    valid = sandbox.validate_target_path(str(tmp_path / "notes" / "a.txt"))
    assert str(valid).endswith("a.txt")


def test_validate_blocks_path_outside_root(tmp_path: Path) -> None:
    sandbox = FilesystemSandbox([str(tmp_path)])
    with pytest.raises(FilesystemSandboxError):
        sandbox.validate_target_path(str(Path("C:/Windows/system32/evil.txt")))

