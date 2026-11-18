"""Run provenance: hashes, environment record, manifests and run directories.

Every actual run gets its own ``results/runs/<run_id>/`` directory; earlier
runs are never overwritten. Hashes cover raw inputs, derived outputs, the
configuration and the source code that actually executed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
    "seaborn", "PyYAML", "pytest", "nbclient", "jupyterlab",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_record() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def source_hashes(root: str | Path) -> dict[str, str]:
    """SHA256 of every .py file under src/ and scripts/ (uncommitted included)."""
    root = Path(root)
    result: dict[str, str] = {}
    for base in (root / "src", root / "scripts"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def new_run_id(config_sha256: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_sha256[:8]}"


def create_run_dir(root: str | Path, run_id: str) -> Path:
    run_dir = Path(root) / "results" / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists, refusing to overwrite: {run_dir}")
    for sub in ("labels", "tables", "figures", "stability", "sensitivity", "artifacts", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def file_hashes(paths: Iterable[str | Path], root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    result: dict[str, str] = {}
    for path in paths:
        path = Path(path).resolve()
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            # outputs living inside results/runs/<id>/ are keyed from results/
            key = path.relative_to(root / "results").as_posix()
        result[key] = sha256_file(path)
    return result


def build_manifest(
    *,
    run_id: str,
    stage: str,
    status: str,
    config_path: str | Path,
    config_sha256: str,
    root: str | Path,
    input_hashes: dict[str, str],
    output_hashes: dict[str, str],
    stage_summary: dict[str, Any],
    warnings: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    root = Path(root)
    return {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "created_at_utc": utc_now_iso(),
        "protocol_version": "v1",
        "config_path": Path(config_path).relative_to(root).as_posix(),
        "config_sha256": config_sha256,
        "environment": environment_record(),
        "source_code_sha256": source_hashes(root),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "stage_summary": stage_summary,
        "warnings": warnings or [],
        "error": error,
    }
