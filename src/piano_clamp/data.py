"""CSV manifest contract for externally stored music passages."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Mapping

from .paths import ConfigurationError, path_within


MANIFEST_FIELDS = (
    "passage_id",
    "composer",
    "work_title",
    "movement",
    "relative_path",
    "start_bar",
    "end_bar",
    "key",
    "meter",
)

# Optional opaque IDs and provenance fields survive normalization and run
# snapshots so expanded corpora can group related movements and trace assets.
OPTIONAL_MANIFEST_FIELDS = (
    "composition_id",
    "work_id",
    "score_version_id",
    "performance_id",
    "recording_id",
    "alignment_id",
    "audio_origin",
    "source_dataset",
    "license",
)

SCORE_EXTENSIONS = {".mxl", ".musicxml", ".xml", ".mid", ".midi"}
AUDIO_EXTENSIONS = {".wav", ".mp3"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ManifestError(ValueError):
    """Raised when a manifest violates the documented contract."""


def _parse_bar(value: str, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"row {row_number}: {field} must be an integer") from exc
    if parsed < 1:
        raise ManifestError(f"row {row_number}: {field} must be at least 1")
    return parsed


def validate_manifest_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    allowed_extensions: set[str] | None = None,
) -> list[dict[str, str]]:
    """Validate and normalize manifest rows without touching source files."""

    extensions = SCORE_EXTENSIONS if allowed_extensions is None else allowed_extensions
    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for row_number, source in enumerate(rows, start=2):
        missing = [field for field in MANIFEST_FIELDS if field not in source]
        if missing:
            raise ManifestError(f"row {row_number}: missing fields: {', '.join(missing)}")
        row = {
            field: "" if source[field] is None else str(source[field]).strip()
            for field in MANIFEST_FIELDS
        }
        row.update(
            {
                field: "" if source[field] is None else str(source[field]).strip()
                for field in OPTIONAL_MANIFEST_FIELDS
                if field in source
            }
        )
        empty = [field for field in MANIFEST_FIELDS if not row[field]]
        if empty:
            raise ManifestError(f"row {row_number}: empty fields: {', '.join(empty)}")
        passage_id = row["passage_id"]
        if not _SAFE_ID.fullmatch(passage_id):
            raise ManifestError(
                f"row {row_number}: passage_id must use only letters, digits, '.', '_' or '-'"
            )
        if passage_id in seen_ids:
            raise ManifestError(f"row {row_number}: duplicate passage_id {passage_id!r}")
        seen_ids.add(passage_id)

        relative_path = Path(row["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ManifestError(f"row {row_number}: relative_path must stay below MUSIC_DATA_ROOT")
        if relative_path.suffix.lower() not in extensions:
            expected = ", ".join(sorted(extensions))
            raise ManifestError(
                f"row {row_number}: unsupported file extension; expected one of {expected}"
            )

        start = _parse_bar(row["start_bar"], "start_bar", row_number)
        end = _parse_bar(row["end_bar"], "end_bar", row_number)
        if end < start:
            raise ManifestError(f"row {row_number}: end_bar precedes start_bar")
        row["start_bar"] = str(start)
        row["end_bar"] = str(end)
        normalized.append(row)

    if not normalized:
        raise ManifestError("manifest contains no passages")
    return normalized


def read_manifest(
    manifest_path: str | Path,
    *,
    allowed_extensions: set[str] | None = None,
) -> list[dict[str, str]]:
    """Read and validate a UTF-8 CSV manifest."""

    path = Path(manifest_path)
    if not path.is_file():
        raise ManifestError(f"manifest does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = [field for field in MANIFEST_FIELDS if field not in columns]
        if missing:
            raise ManifestError(f"manifest is missing columns: {', '.join(missing)}")
        return validate_manifest_rows(reader, allowed_extensions=allowed_extensions)


def resolve_source_files(
    rows: Iterable[Mapping[str, str]], data_root: str | Path
) -> dict[str, Path]:
    """Resolve and require every manifest source below MUSIC_DATA_ROOT."""

    sources: dict[str, Path] = {}
    for row in rows:
        passage_id = row["passage_id"]
        try:
            source = path_within(
                data_root,
                row["relative_path"],
                field=f"relative_path for {passage_id}",
            )
        except ConfigurationError as exc:
            raise ManifestError(str(exc)) from exc
        if not source.is_file():
            raise ManifestError(f"source file for {passage_id!r} does not exist: {source}")
        sources[passage_id] = source
    return sources


def write_manifest_snapshot(rows: Iterable[Mapping[str, str]], destination: str | Path) -> None:
    """Write the normalized input contract to an external run directory."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    optional = tuple(
        field for field in OPTIONAL_MANIFEST_FIELDS if any(field in row for row in materialized)
    )
    fields = MANIFEST_FIELDS + optional
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields}
            for row in materialized
        )
