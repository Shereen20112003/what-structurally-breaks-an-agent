"""Summarize exploratory model-pilot trials and authoritative usage logs."""

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "results_model_pilot"
EPISODES = PILOT / "episodes.jsonl"


def analyze():
    episodes = [
        json.loads(line)
        for line in EPISODES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    planned = []
    seen_cells = set()
    for episode in episodes:
        key = (episode["factor"], episode["level"], episode["seed"])
        if key in seen_cells:
            continue
        seen_cells.add(key)
        planned.append(episode)
    cells = {}
    for factor in ("depth", "branching", "width", "observation"):
        cells[factor] = {}
        for level in ("low", "high"):
            trials = sorted(
                (episode for episode in planned
                 if episode["factor"] == factor and episode["level"] == level),
                key=lambda episode: episode["seed"],
            )
            cells[factor][level] = {
                "n": len(trials),
                "successes": sum(episode["success"] for episode in trials),
                "success_rate": (
                    mean(episode["success"] for episode in trials)
                    if trials else None
                ),
                "mean_completed": (
                    mean(episode["completed"] for episode in trials)
                    if trials else None
                ),
                "seeds": [episode["seed"] for episode in trials],
                "trial_results": [
                    {
                        "seed": episode["seed"],
                        "success": episode["success"],
                        "completed": episode["completed"],
                        "decisions": episode["decisions"],
                        "failure": episode["failure"],
                    }
                    for episode in trials
                ],
            }

    smoke_path = ROOT / "results" / "pilot_auto_smoke_usage.json"
    usage_files = sorted((PILOT / "usage").glob("call_*.json"))
    usage_reports = [json.loads(path.read_text(encoding="utf-8"))
                     for path in usage_files]
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    total_nano_aiu = smoke["totalNanoAiu"] + sum(
        report["totalNanoAiu"] for report in usage_reports
    )
    premium_cost = smoke.get("totalPremiumRequestCost", 0) + sum(
        report.get("totalPremiumRequestCost", 0) for report in usage_reports
    )
    token_counts = {}
    for direction in ("input", "cache_read", "cache_write", "output"):
        token_counts[direction] = smoke.get("tokenDetails", {}).get(
            direction, {}
        ).get("tokenCount", 0) + sum(
            report.get("tokenDetails", {}).get(direction, {}).get("tokenCount", 0)
            for report in usage_reports
        )

    result = {
        "status": "exploratory_pilot_not_confirmatory",
        "provider": "GitHub Copilot",
        "model": sorted({
            model_id for episode in planned
            for model_id in episode.get("model_ids_seen", [])
        }),
        "unique_analyzed_episode_count": len(planned),
        "excluded_duplicate_episode_count": len(episodes) - len(planned),
        "task_count_per_factor_level": 3,
        "operations_per_task": 8,
        "cells": cells,
        "usage_call_count_including_smoke": len(usage_reports) + 1,
        "premium_request_cost_including_smoke": premium_cost,
        "total_nano_aiu_including_smoke": total_nano_aiu,
        "total_ai_credits_including_smoke": total_nano_aiu / 1_000_000_000,
        "tokens_including_smoke": token_counts,
        "limitations": [
            "One routed model only; no Gemini or Qwen model was available in this Copilot CLI account.",
            "No reasoning-budget comparison: the available Auto route does not accept a selectable reasoning-effort setting.",
            "Three task seeds per cell are exploratory and do not support reliable inferential claims.",
            "The tested workflow family is synthetic chains only; there are no external-task results.",
            "The intervention changes task properties but does not estimate a general causal effect across models or task domains.",
        ],
    }
    summary = {
        "status": "exploratory_pilot_not_confirmatory",
        "protocol": json.loads(
            (ROOT / "model_pilot_protocol.json").read_text(encoding="utf-8")
        ),
        "extension_protocol": json.loads(
            (ROOT / "model_pilot_extension_protocol.json").read_text(
                encoding="utf-8"
            )
        ),
        "recorded_episode_count_including_recovery_duplicates": len(episodes),
        "unique_analyzed_episode_count": len(planned),
        "excluded_duplicate_episode_count": len(episodes) - len(planned),
        "episodes": episodes,
        "model_ids_seen": result["model"],
        "usage_call_count_including_smoke": len(usage_reports) + 1,
        "premium_request_cost_including_smoke": premium_cost,
        "total_nano_aiu_including_smoke": total_nano_aiu,
        "total_ai_credits_including_smoke":
            total_nano_aiu / 1_000_000_000,
        "token_counts_including_smoke": token_counts,
        "stop_reason": "exploratory pilot complete; no further inference authorized",
    }
    (PILOT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (PILOT / "analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Intervention & Level & Success & Mean completed & $n$ \\",
        r"\midrule",
    ]
    labels = {
        "depth": "Depth",
        "branching": "Branching",
        "width": "Register width",
        "observation": "Feedback",
    }
    level_labels = {
        "depth": {"low": "1", "high": "8"},
        "branching": {"low": "2", "high": "8"},
        "width": {"low": "1", "high": "8"},
        "observation": {"low": "History", "high": "Snapshot"},
    }
    for factor in labels:
        for level in ("low", "high"):
            cell = cells[factor][level]
            completed = cell["mean_completed"]
            lines.append(
                f"{labels[factor]} & {level_labels[factor][level]} & "
                f"{cell['successes']}/{cell['n']} & "
                f"{completed:.2f} & {cell['n']} " + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (PILOT / "pilot_summary.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in ("cells", "limitations")
    }, indent=2))


if __name__ == "__main__":
    analyze()
