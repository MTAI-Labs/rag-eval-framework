"""Tiny JSON-over-HTTP client (stdlib ``urllib``).

Keeps the core dependency-free: the adapters need POST-JSON-get-JSON and
nothing else, and the CI smoke path must not require wheels to be installed.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import time
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


#: Transport failures worth retrying. A long ingest holds one connection open
#: for many minutes, and the far end closing it is common enough that dying on
#: the first occurrence loses hours of queued work.
TRANSIENT = (
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    ConnectionResetError,
    TimeoutError,
    socket.timeout,
)


def post_multipart(
    url: str,
    body: bytes,
    content_type: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 600.0,
    retries: int = 3,
    backoff: float = 5.0,
) -> dict[str, Any]:
    """POST a prebuilt multipart body, retrying transient transport failures.

    Ingestion is slow, hence the long default timeout. The retry exists because
    a dropped connection is not evidence the work failed -- it may well have
    been accepted -- and abandoning a whole run over one is worse than trying
    again.
    """
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", content_type)
        request.add_header("Accept", "application/json")
        for key, value in (headers or {}).items():
            if value:
                request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8")
            return json.loads(text) if text.strip() else {}
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            raise HttpError(exc.code, exc.read().decode("utf-8", "replace"), url) from exc
        except TRANSIENT as exc:  # pragma: no cover - network path
            last = exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            if not isinstance(exc.reason, TRANSIENT):
                raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
            last = exc
        if attempt < retries:
            time.sleep(backoff * attempt)
    raise RuntimeError(
        f"{url}: transport failed after {retries} attempt(s): {type(last).__name__}: {last}"
    )


def post_stream(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> tuple[list[dict[str, Any]], str]:
    """POST JSON and return ``(events, content_type)``.

    Some RAG servers reply with Server-Sent Events regardless of
    ``"stream": false`` in the request, so the caller cannot know in advance
    which shape it will get. This returns the parsed ``data:`` payloads when
    the response is an event stream, and a single-element list holding the
    decoded body when it is ordinary JSON -- leaving the decision to the
    caller, which can see the content type.
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json, text/event-stream")
    for key, value in (headers or {}).items():
        if value:
            request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raise HttpError(exc.code, exc.read().decode("utf-8", "replace"), url) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc

    if "text/event-stream" not in content_type:
        return ([json.loads(body)] if body.strip() else []), content_type
    return parse_sse(body), content_type


def parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse an SSE body into its ``data:`` payloads.

    Comments (``:`` keep-alives), the ``[DONE]`` sentinel and any payload that
    is not a JSON object are skipped rather than raising -- one malformed event
    in a long stream should not discard the rest of the answer.
    """
    events: list[dict[str, Any]] = []
    for block in re.split(r"\r?\n\r?\n", body):
        payload = "".join(
            line[5:].lstrip() if line.startswith("data:") else ""
            for line in block.splitlines()
            if line.startswith("data:")
        )
        if not payload or payload.strip() == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


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
