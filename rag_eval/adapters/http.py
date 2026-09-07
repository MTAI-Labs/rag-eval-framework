"""Tiny JSON-over-HTTP client (stdlib ``urllib``).

Keeps the core dependency-free: the adapters need POST-JSON-get-JSON and
nothing else, and the CI smoke path must not require wheels to be installed.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"HTTP {status} from {url}: {body[:400]}")
        self.status = status
        self.body = body
        self.url = url


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    for key, value in (headers or {}).items():
        if value:
            request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raise HttpError(exc.code, exc.read().decode("utf-8", "replace"), url) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
    return json.loads(body) if body.strip() else {}


def post_multipart(
    url: str,
    body: bytes,
    content_type: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """POST a prebuilt multipart body. Ingestion is slow, hence the long default."""
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    request.add_header("Accept", "application/json")
    for key, value in (headers or {}).items():
        if value:
            request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raise HttpError(exc.code, exc.read().decode("utf-8", "replace"), url) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
    return json.loads(text) if text.strip() else {}


def get_json(
    url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0
) -> Any:
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "application/json")
    for key, value in (headers or {}).items():
        if value:
            request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raise HttpError(exc.code, exc.read().decode("utf-8", "replace"), url) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
    return json.loads(body) if body.strip() else None
