"""Golden-set JSONL loader with schema + checksum validation.

The golden set is versioned data that gates every score in this framework, so
a change to it must be reviewable: ``golden_v1.jsonl`` ships next to
``golden_v1.manifest.json`` recording a sha256, the row count and the provenance
of the conversion. The loader refuses a dataset whose bytes no longer match
its manifest unless the caller explicitly opts out.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from rag_eval.types import GoldenItem

REQUIRED_FIELDS = ("id", "question", "expected_answer")


class DatasetError(ValueError):
    """Schema, checksum or manifest problem in a golden set."""


@dataclass
class DatasetManifest:
    path: str
    sha256: str
    count: int
    created_at: str
    source: str = ""
    source_sha256: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "count": self.count,
            "created_at": self.created_at,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "notes": self.notes,
        }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_checksum(path: str | Path) -> str:
    """Checksum of the dataset file exactly as it sits on disk."""
    return sha256_file(path)


def manifest_path_for(dataset_path: str | Path) -> Path:
    p = Path(dataset_path)
    return p.with_suffix("").with_suffix(".manifest.json") if p.suffix == ".jsonl" \
        else p.with_name(p.name + ".manifest.json")


def read_manifest(dataset_path: str | Path) -> DatasetManifest | None:
    mp = manifest_path_for(dataset_path)
    if not mp.exists():
        return None
    raw = json.loads(mp.read_text(encoding="utf-8"))
    return DatasetManifest(
        path=raw.get("path", Path(dataset_path).name),
        sha256=raw["sha256"],
        count=int(raw["count"]),
        created_at=raw.get("created_at", ""),
        source=raw.get("source", ""),
        source_sha256=raw.get("source_sha256", ""),
        notes=raw.get("notes", ""),
    )


def validate_item(raw: dict[str, Any], line_no: int) -> GoldenItem:
    missing = [f for f in REQUIRED_FIELDS if not str(raw.get(f, "")).strip()]
    if missing:
        raise DatasetError(f"line {line_no}: missing required field(s): {', '.join(missing)}")
    ref = raw.get("reference")
    if ref is not None and not isinstance(ref, dict):
        raise DatasetError(f"line {line_no}: 'reference' must be an object or null")
    if isinstance(ref, dict):
        for f in ("doc_type", "sitting_date"):
            if not ref.get(f):
                raise DatasetError(f"line {line_no}: reference is missing '{f}'")
    try:
        return GoldenItem.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError(f"line {line_no}: {exc}") from exc


def iter_golden_set(path: str | Path) -> Iterator[GoldenItem]:
    """Stream a golden set, validating each row's schema."""
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"line {line_no}: invalid JSON: {exc}") from exc
            yield validate_item(raw, line_no)


def load_golden_set(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    limit: int | None = None,
    ids: Sequence[str] | None = None,
) -> list[GoldenItem]:
    """Load, validate and return the golden set.

    ``verify_checksum`` compares the file against ``*.manifest.json`` and
    raises if they diverge -- an edited golden set silently changing scores is
    exactly the failure this framework exists to prevent. Pass ``False`` only
    for a deliberately ad-hoc slice.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"golden set not found: {path}")

    items = list(iter_golden_set(path))

    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise DatasetError(f"duplicate question id: {item.id}")
        seen.add(item.id)

    if verify_checksum:
        manifest = read_manifest(path)
        if manifest is None:
            raise DatasetError(
                f"no manifest beside {path.name}; regenerate with "
                f"'rag-eval convert-golden' or pass --no-verify-checksum"
            )
        actual = dataset_checksum(path)
        if actual != manifest.sha256:
            raise DatasetError(
                f"{path.name} does not match its manifest (sha256 {actual[:12]}… vs "
                f"{manifest.sha256[:12]}…). The golden set changed without review."
            )
        if manifest.count != len(items):
            raise DatasetError(
                f"{path.name} has {len(items)} rows but its manifest records {manifest.count}"
            )

    if ids:
        wanted = set(ids)
        items = [i for i in items if i.id in wanted]
        unknown = wanted - {i.id for i in items}
        if unknown:
            raise DatasetError(f"unknown question id(s): {', '.join(sorted(unknown))}")
    if limit is not None:
        items = items[:limit]
    return items


def write_golden_set(
    path: str | Path,
    items: Iterable[GoldenItem],
    *,
    source: str = "",
    source_sha256: str = "",
    notes: str = "",
) -> DatasetManifest:
    """Write a golden set plus its manifest, and return the manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(items)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    manifest = DatasetManifest(
        path=path.name,
        sha256=dataset_checksum(path),
        count=len(items),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=source,
        source_sha256=source_sha256,
        notes=notes,
    )
    manifest_path_for(path).write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
