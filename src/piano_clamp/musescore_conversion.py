"""Standalone MuseScore-to-MusicXML conversion ahead of the core pipeline."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .embedding_io import sha256_path
from .features import FeatureError, extract_musicxml_features


MUSESCORE_SOURCE_EXTENSIONS = {".mscx", ".mscz"}
REPORT_FIELDS = (
    "source_path",
    "output_path",
    "status",
    "error",
    "musescore_version",
    "source_sha256",
    "output_sha256",
    "source_format",
    "output_format",
)


class MuseScoreConversionError(RuntimeError):
    """Raised when the standalone conversion stage cannot run safely."""


@dataclass(frozen=True)
class ConversionRecord:
    """One source-file conversion outcome."""

    source_path: str
    output_path: str
    status: str
    error: str
    musescore_version: str
    source_sha256: str
    output_sha256: str
    source_format: str
    output_format: str


@dataclass(frozen=True)
class ConversionSummary:
    """Summary and report locations for a conversion run."""

    source_count: int
    converted_count: int
    failed_count: int
    skipped_count: int
    musescore_version: str
    csv_report: Path
    json_report: Path


def discover_musescore_sources(input_root: str | Path) -> list[Path]:
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise MuseScoreConversionError(f"input root does not exist: {root}")
    discovered = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MUSESCORE_SOURCE_EXTENSIONS
    ]
    return sorted(discovered)


def probe_musescore_version(binary: str) -> str:
    try:
        result = subprocess.run(
            [binary, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MuseScoreConversionError(f"MuseScore CLI is not available: {binary}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stdout or exc.stderr or "").strip()
        raise MuseScoreConversionError(
            f"cannot determine MuseScore version from {binary}: {detail or exc}"
        ) from exc
    version = (result.stdout or result.stderr).strip().splitlines()
    if not version:
        raise MuseScoreConversionError(f"{binary} --version produced no output")
    return version[0].strip()


def output_path_for_source(source: Path, *, input_root: Path, output_root: Path) -> Path:
    relative = source.relative_to(input_root)
    return (output_root / relative).with_suffix(".musicxml")


def validate_exported_musicxml(path: str | Path) -> None:
    try:
        extract_musicxml_features(path)
    except FeatureError as exc:
        raise MuseScoreConversionError(str(exc)) from exc


def _convert_one(
    source: Path,
    *,
    input_root: Path,
    output_root: Path,
    binary: str,
    musescore_version: str,
    overwrite: bool,
) -> ConversionRecord:
    output = output_path_for_source(source, input_root=input_root, output_root=output_root)
    source_sha256 = sha256_path(source)
    if output.exists() and not overwrite:
        return ConversionRecord(
            source_path=str(source),
            output_path=str(output),
            status="skipped_existing",
            error="output already exists; rerun with --overwrite to replace it",
            musescore_version=musescore_version,
            source_sha256=source_sha256,
            output_sha256=sha256_path(output),
            source_format=source.suffix.lower().lstrip("."),
            output_format=output.suffix.lower().lstrip("."),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [binary, "-o", str(output), str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        if not output.is_file():
            raise MuseScoreConversionError("MuseScore returned success but wrote no output file")
        validate_exported_musicxml(output)
    except (MuseScoreConversionError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stdout or exc.stderr or "").strip()
        message = str(exc)
        if detail and detail not in message:
            message = f"{message}: {detail}"
        output_sha256 = sha256_path(output) if output.is_file() else ""
        return ConversionRecord(
            source_path=str(source),
            output_path=str(output),
            status="failed",
            error=message,
            musescore_version=musescore_version,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
            source_format=source.suffix.lower().lstrip("."),
            output_format=output.suffix.lower().lstrip("."),
        )
    if result.stderr.strip():
        error = result.stderr.strip()
    else:
        error = ""
    return ConversionRecord(
        source_path=str(source),
        output_path=str(output),
        status="success",
        error=error,
        musescore_version=musescore_version,
        source_sha256=source_sha256,
        output_sha256=sha256_path(output),
        source_format=source.suffix.lower().lstrip("."),
        output_format=output.suffix.lower().lstrip("."),
    )


def write_conversion_reports(
    records: list[ConversionRecord],
    *,
    csv_path: str | Path,
    json_path: str | Path,
    musescore_version: str,
) -> ConversionSummary:
    csv_output = Path(csv_path)
    json_output = Path(json_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    summary = {
        "schema_version": 1,
        "musescore_version": musescore_version,
        "source_count": len(records),
        "converted_count": sum(record.status == "success" for record in records),
        "failed_count": sum(record.status == "failed" for record in records),
        "skipped_count": sum(record.status == "skipped_existing" for record in records),
        "records": [asdict(record) for record in records],
    }
    json_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ConversionSummary(
        source_count=int(summary["source_count"]),
        converted_count=int(summary["converted_count"]),
        failed_count=int(summary["failed_count"]),
        skipped_count=int(summary["skipped_count"]),
        musescore_version=musescore_version,
        csv_report=csv_output,
        json_report=json_output,
    )


def convert_musescore_scores(
    *,
    input_root: str | Path,
    output_root: str | Path,
    report_root: str | Path,
    musescore_binary: str = "mscore",
    overwrite: bool = False,
) -> ConversionSummary:
    input_path = Path(input_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    report_path = Path(report_root).expanduser().resolve()
    sources = discover_musescore_sources(input_path)
    version = probe_musescore_version(musescore_binary)
    records = [
        _convert_one(
            source,
            input_root=input_path,
            output_root=output_path,
            binary=musescore_binary,
            musescore_version=version,
            overwrite=overwrite,
        )
        for source in sources
    ]
    return write_conversion_reports(
        records,
        csv_path=report_path / "musescore_conversion_report.csv",
        json_path=report_path / "musescore_conversion_report.json",
        musescore_version=version,
    )


def summary_dict(summary: ConversionSummary) -> dict[str, Any]:
    """Return a JSON-serializable summary."""

    return {
        "source_count": summary.source_count,
        "converted_count": summary.converted_count,
        "failed_count": summary.failed_count,
        "skipped_count": summary.skipped_count,
        "musescore_version": summary.musescore_version,
        "csv_report": str(summary.csv_report),
        "json_report": str(summary.json_report),
    }
