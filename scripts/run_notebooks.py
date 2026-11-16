"""Execute the research notebooks top-to-bottom with nbclient (no interaction).

Usage:
    python scripts/run_notebooks.py            # both notebooks
    python scripts/run_notebooks.py 01         # only notebook 01

Executed copies (with outputs) are saved under notebooks/executed/ so the
source notebooks stay clean. Requires a completed prepare/eda/cluster run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "01": "01_data_audit_and_eda.ipynb",
    "02": "02_clustering_experiments.ipynb",
}


def main() -> int:
    keys = sys.argv[1:] or list(NOTEBOOKS)
    out_dir = ROOT / "notebooks" / "executed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        if key not in NOTEBOOKS:
            print(f"unknown notebook key {key!r}; choose from {list(NOTEBOOKS)}", file=sys.stderr)
            return 2
        path = ROOT / "notebooks" / NOTEBOOKS[key]
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook, timeout=1200, kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
        target = out_dir / NOTEBOOKS[key]
        nbformat.write(notebook, target)
        print(f"[{key}] executed OK -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
