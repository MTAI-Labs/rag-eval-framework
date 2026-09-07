"""multipart/form-data encoding (stdlib).

``POST /documents`` on the NVIDIA ingest server takes files plus a JSON ``data``
field as multipart, which ``urllib`` cannot build on its own. Forty lines here
keeps the core dependency-free rather than pulling in ``requests`` for one call.
"""

from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import Iterable, Sequence


def encode(
    fields: Sequence[tuple[str, str]],
    files: Iterable[tuple[str, Path]],
) -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for the given form fields and files."""
    boundary = f"----ragEval{uuid.uuid4().hex}"
    marker = f"--{boundary}".encode()
    parts: list[bytes] = []

    for name, value in fields:
        parts += [
            marker,
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            str(value).encode("utf-8"),
        ]

    for name, path in files:
        path = Path(path)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts += [
            marker,
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"'.encode(),
            f"Content-Type: {ctype}".encode(),
            b"",
            path.read_bytes(),
        ]

    parts += [f"--{boundary}--".encode(), b""]
    return b"\r\n".join(parts), f"multipart/form-data; boundary={boundary}"
