"""A stub RAG service the adapter contract suite runs every adapter against.

Real HTTP on a loopback port rather than monkeypatched transport: the contract
includes how an adapter behaves when a connection drops or a body is truncated,
and that cannot be tested by replacing the function that would have failed.

One server answers in both wire shapes -- the NVIDIA Blueprint's
``/v1/generate`` and TanyaParlimen's ``/api/ask`` -- so the same assertions run
against each adapter without the test knowing which is which.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

#: Behaviours a test can switch the service into.
OK = "ok"
HTTP_ERROR = "http_error"          # 500 with a body
MALFORMED = "malformed"            # 200 with a body that is not JSON
EMPTY = "empty"                    # 200 with nothing retrieved
DROP = "drop"                      # close the connection without responding
SSE = "sse"                        # event-stream instead of JSON

_NVIDIA_BODY = {
    "choices": [{"message": {"content": "Pandan dan Setiawangsa."}}],
    "citations": {"total_results": 2, "results": [
        {"content": "…Pandan dan Setiawangsa…", "score": 0.81, "document_id": "d1",
         "document_name": "dr_2026-06-22.pdf",
         "metadata": {"dewan": "dewan rakyat", "session_date": "2026-06-22",
                      "page_number": 9}},
        {"content": "…second chunk…", "score": 0.42, "document_id": "d2",
         "document_name": "dn_2026-08-04.pdf",
         "metadata": {"dewan": "dewan negara", "session_date": "2026-08-04",
                      "page_number": 15}},
    ]},
    "usage": {"prompt_tokens": 900, "completion_tokens": 40},
    "model": "test-llm",
}

_TANYA_BODY = {
    "answer": "Pandan dan Setiawangsa.",
    "sources": [
        {"text": "…Pandan dan Setiawangsa…", "score": 0.81, "id": "d1",
         "sitting_id": "dr_2026-06-22", "page": 9},
        {"text": "…second chunk…", "score": 0.42, "id": "d2",
         "sitting_id": "dn_2026-08-04", "page": 15},
    ],
    "usage": {"prompt_tokens": 900, "completion_tokens": 40},
    "model": "test-llm",
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass

    @property
    def mode(self) -> str:
        return self.server.mode  # type: ignore[attr-defined]

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("health"):
            self._send(200, json.dumps({"message": "Service is up."}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.server.requests.append(self.path)  # type: ignore[attr-defined]

        if self.mode == DROP:
            self.close_connection = True
            return
        if self.mode == HTTP_ERROR:
            self._send(500, b'{"detail":"upstream exploded"}')
            return
        if self.mode == MALFORMED:
            self._send(200, b"<html>not json at all</html>")
            return

        nvidia = "/generate" in self.path or "/search" in self.path
        if self.mode == EMPTY:
            body = ({"choices": [{"message": {"content": ""}}],
                     "citations": {"total_results": 0, "results": []}}
                    if nvidia else {"answer": "", "sources": []})
            self._send(200, json.dumps(body).encode())
            return
        if self.mode == SSE and nvidia:
            events = [
                {"choices": [{"delta": {"content": "Pandan dan Setiawangsa."}}],
                 "citations": _NVIDIA_BODY["citations"], "model": "test-llm",
                 "object": "chat.completion.chunk"},
                {"choices": [{"delta": {"content": ""}}],
                 "citations": {"total_results": 0, "results": []},
                 "usage": _NVIDIA_BODY["usage"], "object": "chat.completion.chunk"},
            ]
            payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            self._send(200, payload.encode(), "text/event-stream; charset=utf-8")
            return

        self._send(200, json.dumps(_NVIDIA_BODY if nvidia else _TANYA_BODY).encode())


class MockRagService:
    """Context manager yielding a running service; ``mode`` switches behaviour."""

    def __init__(self, mode: str = OK) -> None:
        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._server.mode = mode          # type: ignore[attr-defined]
        self._server.requests = []        # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def requests(self) -> list[str]:
        return self._server.requests  # type: ignore[attr-defined]

    def set_mode(self, mode: str) -> None:
        self._server.mode = mode  # type: ignore[attr-defined]

    def __enter__(self) -> "MockRagService":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
