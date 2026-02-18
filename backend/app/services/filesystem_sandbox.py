from pathlib import Path


class FilesystemSandboxError(ValueError):
    pass


class FilesystemSandbox:
    def __init__(self, allowed_roots: list[str]):
        self._roots = [Path(root).resolve() for root in allowed_roots if root.strip()]
        if not self._roots:
            raise FilesystemSandboxError("No allowed write roots configured")

    def validate_target_path(self, target_path: str) -> Path:
        raw = Path(target_path)
        resolved = raw.resolve()

        if resolved.exists() and resolved.is_symlink():
            raise FilesystemSandboxError("Symlink targets are not allowed")

        if not any(self._is_within_root(resolved, root) for root in self._roots):
            raise FilesystemSandboxError("Target path is outside allowed write roots")

        return resolved

    @staticmethod
    def _is_within_root(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

