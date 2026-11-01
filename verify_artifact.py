"""Fail loudly if stored evidence and compiled deliverables disagree."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from analyze_local_study import analyze_episodes
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent


def verify():
    metadata = json.loads((ROOT / "results" / "metadata.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Experiment source changed since run: {name}")
    episodes = ROOT / "results" / "episodes.csv"
    if hashlib.sha256(episodes.read_bytes()).hexdigest() != (
            metadata["episodes_sha256"]):
        raise RuntimeError("Episode file hash mismatch")
    counts, exact_failures = Counter(), 0
    conditional = []
    with episodes.open(newline="") as stream:
        for row in csv.DictReader(stream):
            counts[row["policy"]] += 1
            if row["policy"] == "exact":
                exact_failures += int(row["success"] != "1")
                if row["calls"] != "8":
                    raise RuntimeError("Successful exact episode has wrong length")
            if (row["family"], row["policy"], row["width"],
                    row["observation"]) == ("chains", "memory2", "8", "history"):
                conditional.append(int(row["success"]))
    if (len(counts) != 8 or set(counts.values()) != {metadata["tasks"]}
            or sum(counts.values()) != metadata["episodes"] or exact_failures):
        raise RuntimeError("Incomplete or invalid episode evidence")
    if protocol["seeds"] != metadata["seeds"]:
        raise RuntimeError("Protocol/run seed count mismatch")
    analysis = json.loads((ROOT / "results" / "analysis.json").read_text())
    if hashlib.sha256((ROOT / "analyze.py").read_bytes()).hexdigest() != (
            analysis["analysis_sha256"]):
        raise RuntimeError("Analysis source changed since analysis")
    documents = {}
    for name in ("paper", "workshop", "review", "supplement"):
        reader = PdfReader(ROOT / f"{name}.pdf")
        pages = [page.extract_text() for page in reader.pages]
        if not all(page.strip() for page in pages):
            raise RuntimeError(f"Empty PDF page in {name}")
        text = "\n".join(pages)
        if "??" in text or "\ufffd" in text:
            raise RuntimeError(f"Unresolved reference or replacement glyph in {name}")
        if name in ("workshop", "review"):
            if "Anonymous Author" not in text or "Shereen Anand" in text or (
                    "t-sheanand@microsoft.com" in text):
                raise RuntimeError(f"Identifying author details found in {name}")
        elif "Shereen Anand" not in text or "t-sheanand@microsoft.com" not in text:
            raise RuntimeError(f"Author metadata missing in {name}")
        documents[name] = {
            "pages": len(pages),
            "sha256": hashlib.sha256((ROOT / f"{name}.pdf").read_bytes()).hexdigest(),
            "references_page": next(
                (i + 1 for i, page in enumerate(pages)
                 if "REFERENCES" in page), None),
        }
        (ROOT / "results" / f"{name}_text.txt").write_text(
            "\n\n".join(f"PAGE {i+1}\n{page}" for i, page in enumerate(pages)),
            encoding="utf-8")
    if documents["workshop"]["references_page"] is None:
        raise RuntimeError("Workshop PDF has no references")
    pilot_analysis = json.loads(
        (ROOT / "results_model_pilot" / "analysis.json").read_text())
    pilot_summary = json.loads(
        (ROOT / "results_model_pilot" / "summary.json").read_text())
    if (pilot_analysis["unique_analyzed_episode_count"] != 24
            or pilot_analysis["excluded_duplicate_episode_count"] != 5
            or pilot_analysis["usage_call_count_including_smoke"] != 168
            or pilot_summary["unique_analyzed_episode_count"] != 24):
        raise RuntimeError("Model-pilot evidence summary is inconsistent")
    factorial_protocol_path = ROOT / "model_factorial_protocol.json"
    factorial_protocol = json.loads(
        factorial_protocol_path.read_text(encoding="utf-8"))
    factorial_episodes_path = (
        ROOT / "results_model_factorial" / "episodes.jsonl")
    factorial_episodes = [
        json.loads(line)
        for line in factorial_episodes_path.read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    factorial_analysis_path = (
        ROOT / "results_model_factorial" / "analysis.json")
    factorial_analysis = json.loads(
        factorial_analysis_path.read_text(encoding="utf-8"))
    factorial_keys = {
        (row["condition_id"], row["seed"]) for row in factorial_episodes}
    expected_factorial_keys = {
        (condition_id, seed)
        for condition_id in {
            row["condition_id"] for row in factorial_episodes}
        for seed in factorial_protocol["seeds"]
    }
    if (len(factorial_episodes) != 16
            or len(factorial_keys) != 16
            or factorial_keys != expected_factorial_keys):
        raise RuntimeError("Copilot fractional-factorial grid is incomplete")
    if (hashlib.sha256(factorial_episodes_path.read_bytes()).hexdigest()
            != factorial_analysis["episode_sha256"]
            or hashlib.sha256(factorial_protocol_path.read_bytes()).hexdigest()
            != factorial_analysis["protocol_sha256"]
            or hashlib.sha256(
                (ROOT / "analyze_model_factorial.py").read_bytes()
            ).hexdigest() != factorial_analysis["analysis_sha256"]):
        raise RuntimeError("Copilot fractional-factorial hash mismatch")
    if (factorial_analysis["episode_count"] != 16
            or factorial_analysis["success_count"] != sum(
                row["success"] for row in factorial_episodes)
            or factorial_analysis["completed_operation_total"] != sum(
                row["completed"] for row in factorial_episodes)
            or factorial_analysis["model_ids_seen"] != ["gpt-6-luna"]):
        raise RuntimeError("Copilot fractional-factorial summary mismatch")
    local_protocol_path = ROOT / "local_model_protocol.json"
    local_protocol = json.loads(local_protocol_path.read_text(encoding="utf-8"))
    local_episodes_path = ROOT / "results_local_model" / "episodes.jsonl"
    local_episodes = [
        json.loads(line)
        for line in local_episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    local_analysis_path = ROOT / "results_local_model" / "analysis.json"
    local_analysis = json.loads(local_analysis_path.read_text(encoding="utf-8"))
    if hashlib.sha256(local_episodes_path.read_bytes()).hexdigest() != (
            local_analysis["episodes_sha256"]):
        raise RuntimeError("Local-model episode file hash mismatch")
    if hashlib.sha256((ROOT / "analyze_local_study.py").read_bytes()).hexdigest() != (
            local_analysis["analysis_source_sha256"]):
        raise RuntimeError("Local-model analysis source hash mismatch")
    if hashlib.sha256(local_protocol_path.read_bytes()).hexdigest() != (
            local_analysis["protocol_sha256"]):
        raise RuntimeError("Local-model protocol hash mismatch")
    if analyze_episodes(local_episodes)["contrasts"] != local_analysis["contrasts"]:
        raise RuntimeError("Local-model analysis does not reproduce")
    if any(
        episode.get("model_file_sha256") != local_protocol["model"]["sha256"]
        or episode.get("runtime", {}).get("archive_sha256")
        != local_protocol["runtime"]["asset_sha256"]
        for episode in local_episodes
    ):
        raise RuntimeError("Local-model/runtime identity differs from protocol")
    attempts_path = ROOT / "results_local_model" / "attempts.jsonl"
    if attempts_path.is_file():
        attempts = [
            json.loads(line)
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(attempt.get("attempt_complete") for attempt in attempts):
            raise RuntimeError("A completed episode was incorrectly stored as interrupted")
    log = (ROOT / "results" / "build.log").read_text(
        encoding="utf-8", errors="strict")
    bad = re.findall(r"(?im)^.*(?:Overfull \\hbox|Missing character|undefined"
                     r"|Citation .* undefined).*$", log)
    if bad:
        raise RuntimeError("Unresolved document warnings:\n" + "\n".join(bad))
    report = {
        "status": "passed", "experiment_sources_match": True,
        "analysis_source_matches": True, "episodes_hash_matches": True,
        "episodes": sum(counts.values()), "exact_failures": exact_failures,
        "model_calls": metadata["model_calls"], "documents": documents,
        "model_pilot": {
            "status": pilot_analysis["status"],
            "unique_analyzed_episodes":
                pilot_analysis["unique_analyzed_episode_count"],
            "excluded_duplicate_episodes":
                pilot_analysis["excluded_duplicate_episode_count"],
            "usage_calls_including_smoke":
                pilot_analysis["usage_call_count_including_smoke"],
            "premium_request_units":
                pilot_analysis["premium_request_cost_including_smoke"],
            "nano_aiu": pilot_analysis["total_nano_aiu_including_smoke"],
        },
        "model_fractional_factorial": {
            "episodes": factorial_analysis["episode_count"],
            "successes": factorial_analysis["success_count"],
            "mean_completed_operations":
                factorial_analysis["mean_completed_operations"],
            "model": factorial_analysis["model"],
            "seed_count": factorial_analysis["seed_count"],
            "new_ai_credits": factorial_analysis["new_factorial_ai_credits"],
            "new_premium_request_units":
                factorial_analysis["new_factorial_premium_request_cost"],
            "total_project_ai_credits":
                factorial_analysis[
                    "total_project_ai_credits_including_smoke"],
        },
        "local_model_study": {
            "episodes": len(local_episodes),
            "model": local_protocol["model"]["repository"],
            "model_sha256": local_protocol["model"]["sha256"],
            "runtime": local_protocol["runtime"]["release"],
            "runtime_archive_sha256": local_protocol["runtime"]["asset_sha256"],
            "paid_calls": 0,
            "failure_counts": local_analysis["failure_counts"],
        },
        "historical_eight_page_target_met":
            documents["workshop"]["references_page"] <= 9,
        "current_workshop_call_verified": False,
        "chains_cache2_width8_history_success": sum(conditional) / len(conditional),
        "document_checks": "No overfull horizontal boxes, unresolved references, or missing glyph warnings",
        "scope": "Local reference-controller experiments, exploratory Copilot pilot and two-seed half-fraction, and one local-model study; authored full draft and anonymized workshop/review PDFs",
    }
    (ROOT / "results" / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
