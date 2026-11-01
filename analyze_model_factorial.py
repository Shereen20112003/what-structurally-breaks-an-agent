"""Validate and summarize the budget-limited model factorial."""

import hashlib
import json
from pathlib import Path

from run_model_factorial import EXPECTED_MODEL, factorial_conditions

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results_model_factorial"
EPISODES_PATH = OUTPUT / "episodes.jsonl"
PROTOCOL_PATH = ROOT / "model_factorial_protocol.json"
FACTORS = ("depth", "branching", "width", "observation")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_episodes():
    return [
        json.loads(line)
        for line in EPISODES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    episodes = read_episodes()
    expected = {
        (condition["condition_id"], seed)
        for condition in factorial_conditions(protocol["condition_order_seed"])
        for seed in protocol["seeds"]
    }
    observed = [(row["condition_id"], row["seed"]) for row in episodes]
    if len(observed) != len(set(observed)):
        raise RuntimeError("Duplicate model-factorial episode")
    if set(observed) != expected:
        raise RuntimeError(
            f"Incomplete model factorial: {len(set(observed))}/"
            f"{len(expected)} planned episodes"
        )
    if any(row.get("status") != "complete" for row in episodes):
        raise RuntimeError("Nonterminal record in completed episode data")
    if any(
        not row.get("model_ids_seen")
        or set(row["model_ids_seen"]) != {EXPECTED_MODEL}
        for row in episodes
    ):
        raise RuntimeError("Unexpected or mixed model route in factorial")
    if any(row.get("decisions") != len(row.get("turns", []))
           for row in episodes):
        raise RuntimeError("Decision/trace length mismatch")
    if any(not 0 <= row["completed"] <= 8 for row in episodes):
        raise RuntimeError("Completed-operation count out of range")

    factor_effects = {}
    for factor in FACTORS:
        per_seed = {}
        for seed in protocol["seeds"]:
            high = [row for row in episodes
                    if row["seed"] == seed and row["codes"][factor] == 1]
            low = [row for row in episodes
                   if row["seed"] == seed and row["codes"][factor] == -1]
            if len(high) != 4 or len(low) != 4:
                raise RuntimeError(f"Unbalanced {factor} contrast for seed {seed}")
            per_seed[str(seed)] = {
                "success_difference_high_minus_low": (
                    sum(row["success"] for row in high) / len(high)
                    - sum(row["success"] for row in low) / len(low)
                ),
                "completed_operations_difference_high_minus_low": (
                    sum(row["completed"] for row in high) / len(high)
                    - sum(row["completed"] for row in low) / len(low)
                ),
            }
        factor_effects[factor] = {
            "seed_paired_differences": per_seed,
            "mean_success_difference": sum(
                item["success_difference_high_minus_low"]
                for item in per_seed.values()
            ) / len(per_seed),
            "mean_completed_operations_difference": sum(
                item["completed_operations_difference_high_minus_low"]
                for item in per_seed.values()
            ) / len(per_seed),
        }

    successes = sum(row["success"] for row in episodes)
    completed = sum(row["completed"] for row in episodes)
    decisions = sum(row["decisions"] for row in episodes)
    failure_counts = {}
    for row in episodes:
        if row["failure"] != "none":
            failure_counts[row["failure"]] = (
                failure_counts.get(row["failure"], 0) + 1
            )
    usage_files = sorted((OUTPUT / "usage").glob("call_*.json"))
    usage_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in usage_files
    ]
    smoke = json.loads(
        (ROOT / "results" / "pilot_auto_smoke_usage.json").read_text(
            encoding="utf-8"
        )
    )
    previous_reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (ROOT / "results_model_pilot" / "usage").glob("call_*.json")
        )
    ]
    total_nano = (
        smoke["totalNanoAiu"]
        + sum(report["totalNanoAiu"] for report in previous_reports)
        + sum(report["totalNanoAiu"] for report in usage_reports)
    )
    premium_cost = (
        smoke.get("totalPremiumRequestCost", 0)
        + sum(report.get("totalPremiumRequestCost", 0)
              for report in previous_reports)
        + sum(report.get("totalPremiumRequestCost", 0)
              for report in usage_reports)
    )

    result = {
        "status": "exploratory_fractional_factorial",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "runner_sha256": sha256(ROOT / "run_model_factorial.py"),
        "analysis_sha256": sha256(Path(__file__)),
        "benchmark_sha256": sha256(ROOT / "benchmark.py"),
        "prompt_source_sha256": sha256(ROOT / "run_model_pilot.py"),
        "episode_sha256": sha256(EPISODES_PATH),
        "episode_count": len(episodes),
        "success_count": successes,
        "success_rate": successes / len(episodes),
        "completed_operation_total": completed,
        "mean_completed_operations": completed / len(episodes),
        "decision_count": decisions,
        "failure_counts": failure_counts,
        "factor_effects": factor_effects,
        "factorial_aliasing": (
            "Main effects are orthogonal but aliased with three-factor "
            "interactions; two-factor interactions are mutually aliased."
        ),
        "seed_count": len(protocol["seeds"]),
        "seed_count_is_confirmatory": False,
        "model": EXPECTED_MODEL,
        "model_ids_seen": sorted({
            model_id for row in episodes for model_id in row["model_ids_seen"]
        }),
        "total_project_ai_credits_including_smoke": total_nano / 1_000_000_000,
        "total_project_premium_request_cost_including_smoke": premium_cost,
        "new_factorial_ai_credits": sum(
            report["totalNanoAiu"] for report in usage_reports
        ) / 1_000_000_000,
        "new_factorial_premium_request_cost": sum(
            report.get("totalPremiumRequestCost", 0)
            for report in usage_reports
        ),
        "limitations": protocol["limitations"],
    }
    (OUTPUT / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    labels = {
        "depth": "Depth (8 vs. 2)",
        "branching": "Branching (8 vs. 2)",
        "width": "Register width (8 vs. 1)",
        "observation": "Snapshot vs. history",
    }
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Factor & Success difference & Operation difference & $n$ per level \\",
        r"\midrule",
    ]
    for factor in FACTORS:
        effect = factor_effects[factor]
        lines.append(
            f"{labels[factor]} & "
            f"{100 * effect['mean_success_difference']:+.1f} pp & "
            f"{effect['mean_completed_operations_difference']:+.2f} & 8 "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ROOT / "generated" / "model_factorial_summary.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in ("factor_effects", "limitations")
    }, indent=2))


if __name__ == "__main__":
    analyze()
