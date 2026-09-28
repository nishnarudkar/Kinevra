"""Artifact storage interface (evidence snapshots, traces, review events).

Keys are relative, slash-separated paths such as "<session_id>/rep0004_peak.jpg". The local
implementation writes under a directory; the AWS one (Phase 7) writes to S3 with the same keys.
"""

from __future__ import annotations

import json
from typing import Any, Protocol


class ArtifactStore(Protocol):
    def put_bytes(self, key: str, data: bytes, content_type: str) -> str:
        """Store `data` under `key`; returns the key."""
        ...

    def get_bytes(self, key: str) -> bytes: ...


def put_json(store: ArtifactStore, key: str, obj: Any) -> str:
    return store.put_bytes(key, json.dumps(obj, indent=2, default=str).encode(), "application/json")


def safe_key(*parts: str) -> str:
    """Join key parts, rejecting absolute paths and '..' (keys come from session ids)."""
    for part in parts:
        if not part or part.startswith(("/", "\\")) or ".." in part.replace("\\", "/").split("/"):
            raise ValueError(f"unsafe key part: {part!r}")
    return "/".join(p.strip("/") for p in parts)
