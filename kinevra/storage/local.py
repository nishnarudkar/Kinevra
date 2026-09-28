"""Local-filesystem artifact store (live mode, tests, local replay)."""

from __future__ import annotations

from pathlib import Path

from kinevra.storage.base import safe_key


class LocalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / safe_key(key)).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"key escapes the store root: {key!r}")
        return path

    def put_bytes(self, key: str, data: bytes, content_type: str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()
