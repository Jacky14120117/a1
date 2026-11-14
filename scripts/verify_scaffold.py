"""Validate this planning handoff; does not validate unexecuted scientific results."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import yaml

from verify_source import audit

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = [
    "docs", "references", "configs", "data/raw", "data/interim", "data/processed",
    "data/metadata", "notebooks", "src/air_quality_clustering", "scripts", "tests",
    "results/runs", "results/figures", "results/tables", "results/labels",
    "results/stability", "reports/templates", "reports/final", "presentations",
    "artifacts", "logs", ".agents/skills/scientific-clustering/agents",
]
FILES = [
    "README.md", "AGENTS.md", "requirements.txt", "configs/project.yaml",
    "docs/00_course_review.md", "docs/01_project_proposal.md",
    "docs/02_requirement_alignment.md", "docs/03_execution_plan.md",
    "docs/04_architecture.md", "docs/05_data_dictionary.md",
    "docs/06_experiment_protocol.md", "docs/07_deliverables.md",
    "docs/08_agent_handoff.md", "references/sources.md",
    "reports/templates/final_report.md", "presentations/storyboard.md",
    ".agents/skills/scientific-clustering/SKILL.md",
    ".agents/skills/scientific-clustering/agents/openai.yaml",
    "data/metadata/source_preflight.json", "scripts/verify_source.py",
    "scripts/verify_scaffold.py", "tests/README.md", "results/README.md",
]


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for relative in DIRECTORIES:
        check("directory:" + relative, (ROOT / relative).is_dir())
    for relative in FILES:
        path = ROOT / relative
        check("file:" + relative, path.is_file() and path.stat().st_size > 0)

    config = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    check("topic_is_3", config["project"]["topic"].startswith("Topic 3:"))
    check("unsupervised_task", config["project"]["task"] == "unsupervised_clustering")
    check("honest_unexecuted_state", config["project"]["model_execution_status"] == "not_started")
    features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    check("exact_six_inputs", config["data"]["features"] == features)
    check("quality_threshold", config["data"]["minimum_valid_hours"] == 18)
    exclusions = config["data"]["excluded_from_clustering"]
    check("excluded_fields_are_strings", all(isinstance(c, str) for c in exclusions))
    check("no_background_in_inputs", not set(features).intersection(exclusions))
    check("two_primary_methods", {"kmeans", "ward"}.issubset(config["algorithms"]))
    check("matched_k_grids", config["algorithms"]["kmeans"]["k"] ==
          config["algorithms"]["ward"]["k"] == list(range(2, 9)))
    check("explicit_kmeans_restarts", config["algorithms"]["kmeans"]["n_init"] == 20)
    check("no_predictive_metrics", config["evaluation"]["predictive_metrics"] == [])
    check("stability_refits_preprocessing", config["stability"]["refit_preprocessing"] is True)

    previous = json.loads((ROOT / "data/metadata/source_preflight.json").read_text(encoding="utf-8"))
    current, payload = audit(ROOT / config["data"]["archive"])
    for key in ["archive_sha256", "selected_csv_sha256", "station_rows", "calendar_days",
                "eligible_complete_days", "excluded_calendar_days", "columns"]:
        check("source:" + key, previous[key] == current[key])
    check("config_matches_source_count", current["eligible_complete_days"] ==
          config["data"]["expected_eligible_days"])
    check("raw_csv_matches_archive", hashlib.sha256((ROOT / config["data"]["csv"]).read_bytes()).hexdigest()
          == hashlib.sha256(payload).hexdigest())

    skill_path = ROOT / ".agents/skills/scientific-clustering/SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8")
    front = re.match(r"\A---\n(.*?)\n---\n", skill_text, flags=re.S)
    skill = yaml.safe_load(front.group(1)) if front else {}
    check("skill_frontmatter", skill.get("name") == "scientific-clustering"
          and bool(skill.get("description")))
    ui = yaml.safe_load((skill_path.parent / "agents/openai.yaml").read_text(encoding="utf-8"))
    check("skill_prompt_invocation", "$scientific-clustering" in ui["interface"]["default_prompt"])

    markdown_files = [ROOT / "README.md", ROOT / "AGENTS.md", skill_path]
    for folder in ["docs", "references", "reports/templates", "presentations", "data",
                   "src/air_quality_clustering", "scripts", "notebooks", "tests", "results"]:
        markdown_files.extend((ROOT / folder).glob("*.md"))
    local_links = 0
    for path in sorted(set(markdown_files)):
        content = path.read_text(encoding="utf-8")
        check("utf8:" + path.relative_to(ROOT).as_posix(), "\ufffd" not in content)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            cleaned = unquote(target.split("#", 1)[0].strip("<>"))
            destination = (path.parent / cleaned).resolve()
            local_links += 1
            check("link:" + path.relative_to(ROOT).as_posix() + " -> " + target,
                  destination.exists())

    check("no_model_runs_created", not any(p.is_dir() for p in (ROOT / "results/runs").iterdir()))
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "planning documents, repository skill, source preflight and directory scaffold",
        "all_checks_passed": all(c["passed"] for c in checks),
        "check_count": len(checks),
        "local_markdown_links_checked": local_links,
        "clustering_executed": False,
        "scientific_results_validated": False,
        "checks": checks,
    }
    output = ROOT / "docs/planning_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["all_checks_passed", "check_count",
          "local_markdown_links_checked", "clustering_executed"]}, indent=2))
    for failed in [c for c in checks if not c["passed"]]:
        print("FAILED:", failed["name"], failed["detail"])
    raise SystemExit(0 if report["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
