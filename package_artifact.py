"""Build explicit source and reproducibility archives without private inputs."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
SOURCES = ["paper.tex", "supplement.tex", "appendix_body.tex"]
INPUTS = [
    "numbers.tex", "curve_data.tex", "audit_table.tex", "summary_table.tex",
    "contrast_table.tex", "scheduler_table.tex", "all_intervals.tex",
    "extended_numbers.tex", "selection_resources.tex", "pairwise_interactions.tex",
    "higher_interactions.tex", "ablation_focus.tex", "ablation_work.tex",
    "ablation_memory.tex", "ablation_observation.tex", "ablation_recovery.tex",
    "ablation_locality.tex", "ablation_selection.tex", "ablation_labels.tex",
    "model_factorial_summary.tex",
    "local_model_summary.tex", "local_model_numbers.tex",
]
CODE = [
    "benchmark.py", "test_benchmark.py", "run_experiments.py", "analyze.py",
    "extended_benchmark.py", "test_extended.py", "run_extended.py",
    "analyze_extended.py", "run_model_pilot.py", "analyze_model_pilot.py",
    "test_model_pilot.py", "model_pilot_protocol.json",
    "model_pilot_extension_protocol.json",
    "run_local_study.py", "analyze_local_study.py", "test_local_study.py",
    "local_model_protocol.json", "run_model_factorial.py",
    "analyze_model_factorial.py", "test_model_factorial.py",
    "model_factorial_protocol.json",
    "protocol.json", "extended_protocol.json", "requirements.txt", "build.ps1",
    "package_artifact.py", "verify_artifact.py",
    "README.txt", "SUBMISSION.txt", ".gitignore",
]
RESULTS = [
    "episodes.csv", "structural_audit.csv", "metadata.json", "analysis.json",
    "policy_summary.csv", "contrasts.csv", "interactions.csv", "curves.csv",
    "failure_counts.csv", "scheduler_validation.csv", "example_trace.json",
    "verification.json", "paper_text.txt", "workshop_text.txt", "review_text.txt",
    "supplement_text.txt", "build.log", "pilot_auto_smoke_usage.json",
]


def package():
    sources = [ROOT / name for name in SOURCES]
    sources += [ROOT / "workshop.tex", ROOT / "review.tex"]
    sources += [ROOT / "generated" / name for name in INPUTS]
    sources += [ROOT / "results_model_pilot" / "pilot_summary.tex"]
    artifact = sources + [ROOT / name for name in CODE]
    artifact += [ROOT / "results" / name for name in RESULTS]
    artifact += [ROOT / "results_extended" / name for name in (
        "episodes.csv", "structural_audit.csv", "metadata.json", "analysis.json",
        "summary.csv", "paired_contrasts.csv", "all_endpoint_interactions.csv",
        "case_design.json")]
    artifact += [ROOT / "results_model_pilot" / name for name in (
        "episodes.jsonl", "summary.json", "analysis.json")]
    artifact += sorted((ROOT / "results_model_pilot" / "usage").glob("call_*.json"))
    artifact += [ROOT / "results_model_factorial" / name for name in (
        "episodes.jsonl", "attempts.jsonl", "analysis.json")]
    artifact += sorted(
        (ROOT / "results_model_factorial" / "usage").glob("call_*.json"))
    artifact += [ROOT / "results_local_model" / name for name in (
        "episodes.jsonl", "smoke.jsonl", "analysis.json")]
    attempts_path = ROOT / "results_local_model" / "attempts.jsonl"
    if attempts_path.is_file():
        artifact.append(attempts_path)
    artifact += [ROOT / "paper.pdf", ROOT / "workshop.pdf",
                 ROOT / "review.pdf", ROOT / "supplement.pdf"]
    for path in artifact:
        if not path.is_file():
            raise FileNotFoundError(f"Required deliverable is missing: {path}")
    manifest = {}
    for archive_name, paths in (("paper-source.zip", sources),
                                ("reproducibility.zip", artifact)):
        with ZipFile(ROOT / archive_name, "w", compression=ZIP_DEFLATED) as archive:
            for path in paths:
                relative = path.relative_to(ROOT).as_posix()
                archive.write(path, relative)
        manifest[archive_name] = {
            "files": len(paths),
            "bytes": (ROOT / archive_name).stat().st_size,
            "sha256": hashlib.sha256(
                (ROOT / archive_name).read_bytes()).hexdigest(),
        }
    (ROOT / "package_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    package()
