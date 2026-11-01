"""Validate and summarize the frozen local-model factorial study."""

import hashlib
import json
from pathlib import Path
import random
import statistics

from run_local_study import FACTORS, OUTPUT, PROTOCOL, SEEDS, factorial_cases

ROOT = Path(__file__).resolve().parent
EPISODES = OUTPUT / "episodes.jsonl"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261101


def expected_cells():
    return {
        (
            seed, family, config.depth, config.branching,
            config.width, config.observation,
        )
        for family, config in factorial_cases()
        for seed in SEEDS
    }


def episode_key(episode):
    cell = episode["factorial_cell"]
    return (
        episode["seed"], cell["family"], cell["depth"],
        cell["branching"], cell["width"], cell["observation"],
    )


def validate_episodes(episodes):
    expected = expected_cells()
    observed = [episode_key(episode) for episode in episodes]
    duplicates = len(observed) - len(set(observed))
    missing = expected - set(observed)
    unexpected = set(observed) - expected
    if duplicates or missing or unexpected:
        raise ValueError(
            "Expected exactly 256 unique preregistered episodes; "
            f"found {len(observed)} rows, {duplicates} duplicates, "
            f"{len(missing)} missing cells, and {len(unexpected)} unexpected cells."
        )
    for episode in episodes:
        cell = episode["factorial_cell"]
        config = episode["config"]
        if any(config[key] != cell[key] for key in FACTORS):
            raise ValueError(f"Episode configuration mismatch in {episode_key(episode)}")
        if config["n"] != 8:
            raise ValueError(f"Episode did not use fixed eight-operation work: {episode_key(episode)}")
        if episode["success"] not in (0, 1):
            raise ValueError(f"Invalid success value in episode {episode_key(episode)}")
        if not 0 <= episode["completed"] <= 8:
            raise ValueError(f"Invalid completed-operation count in {episode_key(episode)}")
        if episode["success"] and episode["completed"] != 8:
            raise ValueError(f"Successful episode did not complete all work: {episode_key(episode)}")
        if len(episode["decisions"]) > 8:
            raise ValueError(f"Episode exceeded the fixed decision limit: {episode_key(episode)}")
    return len(expected)


def quantile(sorted_values, probability):
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def factor_contrast(episodes, factor, rng):
    levels = FACTORS[factor]
    low, high = levels[0], levels[-1]
    by_seed_level = {
        seed: {level: [] for level in levels}
        for seed in SEEDS
    }
    for episode in episodes:
        cell = episode["factorial_cell"]
        by_seed_level[episode["seed"]][cell[factor]].append(episode["success"])
    paired_differences = [
        statistics.mean(by_seed_level[seed][high])
        - statistics.mean(by_seed_level[seed][low])
        for seed in SEEDS
    ]
    samples = [
        statistics.mean(rng.choices(paired_differences, k=len(SEEDS)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    samples.sort()
    return {
        "factor": factor,
        "contrast": f"{high} - {low}",
        "low_level": low,
        "low_successes": sum(
            episode["success"] for episode in episodes
            if episode["factorial_cell"][factor] == low
        ),
        "high_level": high,
        "high_successes": sum(
            episode["success"] for episode in episodes
            if episode["factorial_cell"][factor] == high
        ),
        "episodes_per_level": len(episodes) // len(levels),
        "low_rate": statistics.mean(
            episode["success"] for episode in episodes
            if episode["factorial_cell"][factor] == low
        ),
        "high_rate": statistics.mean(
            episode["success"] for episode in episodes
            if episode["factorial_cell"][factor] == high
        ),
        "paired_difference": statistics.mean(paired_differences),
        "seed_cluster_bootstrap_95_percent_interval": [
            quantile(samples, 0.025),
            quantile(samples, 0.975),
        ],
        "per_seed_paired_differences": paired_differences,
    }


def analyze_episodes(episodes):
    episode_count = validate_episodes(episodes)
    rng = random.Random(BOOTSTRAP_SEED)
    contrasts = [
        factor_contrast(episodes, factor, rng)
        for factor in FACTORS
    ]
    failure_counts = {}
    for episode in episodes:
        failure = episode["first_failure"]
        failure_counts[failure] = failure_counts.get(failure, 0) + 1
    decisions = [
        decision for episode in episodes for decision in episode["decisions"]
    ]
    schema_valid = sum(
        decision["parse_error"] is None for decision in decisions
    )
    return {
        "analysis": "descriptive full-factorial marginal contrasts",
        "confirmatory_claims": False,
        "episode_count": episode_count,
        "success_count": sum(episode["success"] for episode in episodes),
        "overall_success_rate": statistics.mean(
            episode["success"] for episode in episodes
        ),
        "completed_operation_total": sum(
            episode["completed"] for episode in episodes
        ),
        "required_operation_total": episode_count * 8,
        "mean_completed_operations": statistics.mean(
            episode["completed"] for episode in episodes
        ),
        "decision_count": len(decisions),
        "schema_valid_decisions": schema_valid,
        "schema_validity_rate": schema_valid / len(decisions) if decisions else 0,
        "total_prompt_tokens": sum(
            decision["prompt_tokens"] for decision in decisions
        ),
        "total_completion_tokens": sum(
            decision["completion_tokens"] for decision in decisions
        ),
        "median_decision_latency_seconds": (
            statistics.median(
                decision["elapsed_seconds"] for decision in decisions
            ) if decisions else 0
        ),
        "unique_seed_clusters": len(SEEDS),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "failure_counts": failure_counts,
        "contrasts": contrasts,
    }


def write_table(summary, path):
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Contrast & Low success & High success & Difference (95\% interval) \\",
        r"\midrule",
    ]
    for result in summary["contrasts"]:
        low_rate = f'{result["low_successes"]}/{result["episodes_per_level"]}'
        high_rate = f'{result["high_successes"]}/{result["episodes_per_level"]}'
        contrast = (
            f'{result["factor"]} '
            f'({result["high_level"]} $-$ {result["low_level"]})'
        )
        difference = result["paired_difference"] * 100
        lower, upper = result["seed_cluster_bootstrap_95_percent_interval"]
        interval = f'{difference:+.1f} pp ({lower * 100:+.1f}, {upper * 100:+.1f})'
        lines.append(
            f'{contrast} & '
            f'{low_rate} ({result["low_rate"]:.3f}) & '
            f'{high_rate} ({result["high_rate"]:.3f}) & '
            f'{interval} \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    episodes = [
        json.loads(line)
        for line in EPISODES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = analyze_episodes(episodes)
    summary["episodes_sha256"] = hashlib.sha256(
        EPISODES.read_bytes()
    ).hexdigest()
    summary["analysis_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    summary["protocol_sha256"] = hashlib.sha256(
        PROTOCOL.read_bytes()
    ).hexdigest()
    analysis_path = OUTPUT / "analysis.json"
    analysis_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    table_path = ROOT / "generated" / "local_model_summary.tex"
    write_table(summary, table_path)
    numbers = (
        f'\\newcommand{{\\LocalEpisodeCount}}{{{summary["episode_count"]}}}\n'
        f'\\newcommand{{\\LocalSuccessCount}}{{{summary["success_count"]}}}\n'
        f'\\newcommand{{\\LocalOverallSuccessRate}}{{{summary["overall_success_rate"]:.3f}}}\n'
        f'\\newcommand{{\\LocalCompletedOperationCount}}{{{summary["completed_operation_total"]}}}\n'
        f'\\newcommand{{\\LocalRequiredOperationCount}}{{{summary["required_operation_total"]}}}\n'
        f'\\newcommand{{\\LocalMeanCompleted}}{{{summary["mean_completed_operations"]:.2f}}}\n'
        f'\\newcommand{{\\LocalDecisionCount}}{{{summary["decision_count"]}}}\n'
        f'\\newcommand{{\\LocalSchemaValidCount}}{{{summary["schema_valid_decisions"]}}}\n'
        f'\\newcommand{{\\LocalSchemaValidRate}}{{{summary["schema_validity_rate"]:.3f}}}\n'
        f'\\newcommand{{\\LocalMedianLatency}}{{{summary["median_decision_latency_seconds"]:.2f}}}\n'
    )
    (ROOT / "generated" / "local_model_numbers.tex").write_text(
        numbers, encoding="utf-8"
    )
    print(f"Validated {summary['episode_count']} episodes.")
    for result in summary["contrasts"]:
        lower, upper = result["seed_cluster_bootstrap_95_percent_interval"]
        print(
            f"{result['factor']}: {result['paired_difference']:+.3f} "
            f"[{lower:+.3f}, {upper:+.3f}]"
        )
    print(f"Analysis: {analysis_path}")
    print(f"Table: {table_path}")


if __name__ == "__main__":
    main()
