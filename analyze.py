"""Seed-block uncertainty and automatically generated manuscript tables."""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent
DIMENSIONS = {"depth": (1, 8), "branching": (2, 8), "width": (1, 8),
              "observation": ("snapshot", "history")}


def mean(values):
    return statistics.fmean(values)


def interval(values, resamples, rng):
    n = len(values)
    draws = sorted(mean(rng.choices(values, k=n)) for _ in range(resamples))
    return draws[int(.025 * resamples)], draws[int(.975 * resamples)]


def summarize():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    metadata = json.loads((ROOT / "results" / "metadata.json").read_text())
    seed_count = protocol["seeds"]
    with (ROOT / "results" / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = 192 * seed_count * 8
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} episodes; got {len(rows)}")
    records = []
    keys = set()
    for row in rows:
        for name in ("depth", "branching", "width", "seed", "success",
                     "completed", "calls", "evaluations"):
            row[name] = int(row[name])
        row["expected_success"] = float(row["expected_success"])
        key = tuple(row[k] for k in ("family", "depth", "branching", "width",
                                    "observation", "seed", "policy"))
        if key in keys:
            raise RuntimeError("Duplicate episode")
        keys.add(key)
        records.append(row)
    rng = random.Random(protocol["bootstrap_seed"])
    resamples = protocol["bootstrap_resamples"]
    by_policy = defaultdict(list)
    for row in records:
        by_policy[(row["family"], row["policy"])].append(row)
    summaries, contrasts, interactions, curves = [], [], [], []
    for (family, policy), group in sorted(by_policy.items()):
        per_seed = defaultdict(list)
        for row in group:
            per_seed[row["seed"]].append(row["success"])
        blocks = [mean(per_seed[s]) for s in range(seed_count)]
        lo, hi = interval(blocks, resamples, rng)
        summaries.append({
            "family": family, "policy": policy, "success": mean(blocks),
            "ci_low": lo, "ci_high": hi,
            "predicted": mean(r["expected_success"] for r in group),
            "completed": mean(r["completed"] for r in group),
            "evaluations": mean(r["evaluations"] for r in group),
            "episodes": len(group),
        })
        for dimension, (low, high) in DIMENSIONS.items():
            bins = defaultdict(list)
            for row in group:
                bins[(row["seed"], row[dimension])].append(row["success"])
            levels = sorted({row[dimension] for row in group},
                            key=lambda x: str(x))
            for level in levels:
                values = [mean(bins[(s, level)]) for s in range(seed_count)]
                curves.append({"family": family, "policy": policy,
                               "dimension": dimension, "level": level,
                               "success": mean(values)})
            changes = [mean(bins[(s, high)]) - mean(bins[(s, low)])
                       for s in range(seed_count)]
            lo, hi = interval(changes, resamples, rng)
            contrasts.append({"family": family, "policy": policy,
                              "dimension": dimension, "estimate": mean(changes),
                              "ci_low": lo, "ci_high": hi})
        bins = defaultdict(list)
        for row in group:
            bins[(row["seed"], row["width"], row["observation"])].append(
                row["success"])
        changes = []
        for s in range(seed_count):
            changes.append(
                mean(bins[(s, 8, "history")]) - mean(bins[(s, 1, "history")])
                - mean(bins[(s, 8, "snapshot")])
                + mean(bins[(s, 1, "snapshot")]))
        lo, hi = interval(changes, resamples, rng)
        interactions.append({"family": family, "policy": policy,
                             "interaction": "width_by_observation",
                             "estimate": mean(changes),
                             "ci_low": lo, "ci_high": hi})
    for name, data in (("policy_summary", summaries), ("contrasts", contrasts),
                       ("interactions", interactions), ("curves", curves)):
        with (ROOT / "results" / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    failures = defaultdict(int)
    for row in records:
        failures[(row["family"], row["policy"], row["failure"])] += 1
    failure_rows = [{"family": f, "policy": p, "category": c, "count": count}
                    for (f, p, c), count in sorted(failures.items())]
    with (ROOT / "results" / "failure_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(failure_rows[0]))
        writer.writeheader()
        writer.writerows(failure_rows)
    generated = ROOT / "generated"
    generated.mkdir(exist_ok=True)
    label = {"exact": "Exact", "memory2": "Cache-2", "memory4": "Cache-4",
             "scan1": "Scan-1", "scan4": "Scan-4",
             "memory2_scan4": "Cache-2 + Scan-4",
             "random_order": "Random schedule", "uniform": "Uniform candidate"}
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Family & Controller & Success (\%) & 95\% interval & Predicted (\%)\\",
             r"\midrule"]
    for row in summaries:
        lines.append(
            f"{row['family'].capitalize()} & {label[row['policy']]} & "
            f"{100*row['success']:.2f} & "
            f"[{100*row['ci_low']:.2f}, {100*row['ci_high']:.2f}] & "
            f"{100*row['predicted']:.2f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "summary_table.tex").write_text("\n".join(lines))
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Family & Controller & Depth & Candidates & Width & History\\",
             r"\midrule"]
    for family, policy in sorted(by_policy):
        values = [next(r for r in contrasts if r["family"] == family
                       and r["policy"] == policy and r["dimension"] == d)
                  for d in DIMENSIONS]
        lines.append(f"{family.capitalize()} & {label[policy]} & " +
                     " & ".join(f"{100*r['estimate']:.2f}" for r in values)
                     + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "contrast_table.tex").write_text("\n".join(lines))
    lines = [r"\begin{longtable}{lllrrr}", r"\toprule",
             r"Family & Controller & Contrast & Estimate & Low & High\\",
             r"\midrule", r"\endhead"]
    for row in contrasts + interactions:
        name = row.get("dimension", "width_by_observation").replace("_", " ")
        lines.append(
            f"{row['family'].capitalize()} & {label[row['policy']]} & {name} & "
            f"{100*row['estimate']:.2f} & {100*row['ci_low']:.2f} & "
            f"{100*row['ci_high']:.2f}" + r"\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (generated / "all_intervals.tex").write_text("\n".join(lines))
    lines = []
    for family in ("chains", "barriers"):
        for policy in ("memory2", "memory4", "scan1", "scan4", "random_order"):
            for dimension in ("depth", "branching", "width"):
                points = sorted((r for r in curves if r["family"] == family and
                                 r["policy"] == policy and
                                 r["dimension"] == dimension),
                                key=lambda r: int(r["level"]))
                coordinates = " ".join(
                    f"({r['level']},{100*r['success']:.4f})" for r in points)
                lines.append(
                    f"\\expandafter\\def\\csname "
                    f"{family}{policy}{dimension}\\endcsname{{{coordinates}}}")
    (generated / "curve_data.tex").write_text("\n".join(lines))
    macros = [
        f"\\newcommand{{\\TaskCount}}{{{metadata['tasks']:,}}}",
        f"\\newcommand{{\\EpisodeCount}}{{{metadata['episodes']:,}}}",
        f"\\newcommand{{\\SeedCount}}{{{seed_count}}}",
    ]
    for row in interactions:
        if row["family"] == "chains" and row["policy"] == "memory2":
            macros.append(f"\\newcommand{{\\CacheInteraction}}"
                          f"{{{100*row['estimate']:.2f}}}")
    (generated / "numbers.tex").write_text("\n".join(macros))
    with (ROOT / "results" / "structural_audit.csv").open(newline="") as stream:
        audits = list(csv.DictReader(stream))
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Family & $d$ & Edges & Masks & Orders & Ready nodes\\",
             r"\midrule"]
    for row in audits:
        if (row["branching"], row["width"], row["observation"]) != (
                "2", "1", "snapshot"):
            continue
        lines.append(
            f"{row['family'].capitalize()} & {row['depth']} & {row['edges']} & "
            f"{row['reachable_completion_masks']} & {row['linear_extensions']} & "
            f"{float(row['mean_ready_nodes']):.2f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "audit_table.tex").write_text("\n".join(lines))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Family & $d$ & Observed (\%) & Wilson interval & Exact (\%)\\",
             r"\midrule"]
    scheduler_rows = []
    for family in ("chains", "barriers"):
        for depth in (1, 2, 4, 8):
            sample = [r for r in records if r["family"] == family and
                      r["depth"] == depth and r["branching"] == 2 and
                      r["width"] == 1 and r["observation"] == "snapshot" and
                      r["policy"] == "random_order"]
            n = len(sample)
            p = mean(r["success"] for r in sample)
            z = 1.959963984540054
            denominator = 1 + z*z/n
            center = (p + z*z/(2*n)) / denominator
            half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
            prediction = mean(r["expected_success"] for r in sample)
            scheduler_rows.append({
                "family": family, "depth": depth, "trials": n,
                "successes": sum(r["success"] for r in sample),
                "success": p, "wilson_low": center-half,
                "wilson_high": center+half, "predicted": prediction})
            lines.append(
                f"{family.capitalize()} & {depth} & {100*p:.2f} & "
                f"[{100*max(0,center-half):.2f}, {100*min(1,center+half):.2f}] & "
                f"{100*prediction:.4f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "scheduler_table.tex").write_text("\n".join(lines))
    with (ROOT / "results" / "scheduler_validation.csv").open(
            "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scheduler_rows[0]))
        writer.writeheader()
        writer.writerows(scheduler_rows)
    report = {
        "episodes": len(records), "summaries": summaries,
        "contrasts": contrasts, "interactions": interactions,
        "max_abs_prediction_error": max(
            abs(r["success"] - r["predicted"]) for r in summaries),
        "analysis_sha256": hashlib.sha256(
            (ROOT / "analyze.py").read_bytes()).hexdigest(),
        "episode_hash_verified": hashlib.sha256(
            (ROOT / "results" / "episodes.csv").read_bytes()).hexdigest()
        == metadata["episodes_sha256"],
    }
    if not report["episode_hash_verified"]:
        raise RuntimeError("Episode hash does not match run metadata")
    (ROOT / "results" / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"episodes": report["episodes"],
                      "max_abs_prediction_error":
                          report["max_abs_prediction_error"],
                      "summaries": summaries, "interactions": interactions},
                     indent=2))


if __name__ == "__main__":
    summarize()
