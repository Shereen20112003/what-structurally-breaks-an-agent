"""Paired follow-on summaries, direct ablations, and complete endpoint interactions."""

from collections import defaultdict
import csv
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import random
import statistics

from analyze import interval

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results_extended"
LABELS = {
    "exact": "Exact", "cache2": "Cache-2", "cache4": "Cache-4",
    "adaptive2": "Next-use-2", "modulo8": "Residue-8",
    "precomputed_plan": "Precomputed plan", "stale8": "Stale-8",
    "scan1": "Scan-1", "scan2": "Scan-2", "scan4": "Scan-4",
    "scan4_fixed": "Fixed-check-4", "scan1_padded4": "Sham-check-4",
    "indexed": "Indexed", "random_order": "Random schedule",
    "uniform": "Uniform", "uniform_eliminate": "Eliminate",
    "cache2_eliminate": "Cache-2 + eliminate",
    "scan1_eliminate": "Scan-1 + eliminate",
}


def escaped(value):
    return str(value).replace("_", r"\_")


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def band(values, samples, rng):
    if min(values) == max(values):
        return values[0], values[0]
    return interval(values, samples, rng)


def summarize():
    protocol = json.loads((ROOT / "extended_protocol.json").read_text())
    metadata = json.loads((OUTPUT / "metadata.json").read_text())
    source_hash = hashlib.sha256((ROOT / "analyze_extended.py").read_bytes()).hexdigest()
    if hashlib.sha256((OUTPUT / "episodes.csv").read_bytes()).hexdigest() != (
            metadata["episodes_sha256"]):
        raise RuntimeError("Extended episode hash mismatch")
    rng = random.Random(protocol["bootstrap_seed"])
    n, samples = protocol["seeds"], protocol["bootstrap_resamples"]
    seed_ids = list(range(protocol["seed_start"], protocol["seed_start"]+n))
    grouped, case_outcomes = defaultdict(list), defaultdict(dict)
    with (OUTPUT / "episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            for name in ("case_id", "depth", "branching", "width", "n", "period",
                         "shock", "retries", "reverse", "seed", "success",
                         "completed", "calls", "errors", "evaluations", "lookups",
                         "preparation_units", "max_cache_entries", "max_cache_bits",
                         "plan_slots", "snapshot_uses", "dependency_errors",
                         "candidate_errors"):
                row[name] = int(row[name])
            row["predicted"] = float(row["predicted"]) if row["predicted"] else None
            grouped[(row["study"], row["level"], row["policy"])].append(row)
            if row["seed"] in case_outcomes[row["case_id"]]:
                raise RuntimeError("Duplicate follow-on case/seed")
            case_outcomes[row["case_id"]][row["seed"]] = row["success"]
    if len(case_outcomes) != metadata["cases"] or any(
            set(values) != set(seed_ids) for values in case_outcomes.values()):
        raise RuntimeError("Missing follow-on case/seed")
    summaries, blocks = [], {}
    for key, group in sorted(grouped.items()):
        study, level, policy = key
        per_seed = defaultdict(list)
        for row in group:
            per_seed[row["seed"]].append(row["success"])
        values = [statistics.fmean(per_seed[s]) for s in seed_ids]
        blocks[key] = values
        low, high = band(values, samples, rng)
        predictions = [r["predicted"] for r in group if r["predicted"] is not None]
        summaries.append({
            "study": study, "level": level, "policy": policy,
            "success": statistics.fmean(values), "low": low, "high": high,
            "predicted": statistics.fmean(predictions) if predictions else "",
            "predicted_episodes": len(predictions), "episodes": len(group),
            **{name: statistics.fmean(row[name] for row in group) for name in
               ("completed", "calls", "errors", "evaluations", "lookups",
                "preparation_units", "max_cache_entries", "max_cache_bits",
                "plan_slots", "snapshot_uses")},
        })
    write_csv("summary.csv", summaries)
    requested = []
    for period in (0, 1):
        for policy in ("exact", "cache2", "scan4", "random_order"):
            requested.append(("work", f"n16_p{period}", policy,
                              "work", f"n4_p{period}", policy, "N16 minus N4"))
    for period in (0, 4):
        for policy in ("adaptive2", "modulo8", "precomputed_plan", "cache4"):
            requested.append(("memory", f"p{period}", policy,
                              "memory", f"p{period}", "cache2",
                              f"{policy} minus cache2"))
    for period in (0, 1, 2, 4):
        for policy in ("exact", "cache2", "scan4", "uniform"):
            requested.append(("observation", f"p{period}_shock1", policy,
                              "observation", f"p{period}_shock0", policy,
                              "Hidden resampling minus public updates"))
    for period in (0, 1):
        for retries in (0, 2, 8, 56):
            requested.append(("recovery", f"r{retries}_p{period}", "uniform_eliminate",
                              "recovery", f"r{retries}_p{period}", "uniform",
                              "Elimination minus replacement"))
    for updates in ("additive", "static"):
        for policy in ("cache2", "adaptive2", "exact", "stale8"):
            requested.append(("locality", f"clustered_{updates}", policy,
                              "locality", f"balanced_{updates}", policy,
                              "Clustered minus balanced"))
    for queries in ("balanced", "clustered", "concentrated"):
        for policy in ("cache2", "adaptive2", "exact", "stale8"):
            requested.append(("locality", f"{queries}_static", policy,
                              "locality", f"{queries}_additive", policy,
                              "Static minus additive"))
    for b in (2, 4, 8):
        for policy in ("scan1_padded4", "scan4_fixed", "indexed"):
            requested.append(("selection", f"b{b}", policy,
                              "selection", f"b{b}", "scan1",
                              f"{policy} minus scan1"))
    for period in (0, 1):
        for policy in ("exact", "cache2", "scan4", "random_order"):
            requested.append(("labels", f"reverse1_p{period}", policy,
                              "labels", f"reverse0_p{period}", policy,
                              "Reversed minus original identifiers"))
    contrasts = []
    for left_study, left_level, left_policy, right_study, right_level, right_policy, label in requested:
        left, right = (left_study, left_level, left_policy), (
            right_study, right_level, right_policy)
        differences = [a-b for a, b in zip(blocks[left], blocks[right])]
        low, high = band(differences, samples, rng)
        contrasts.append({
            "study": left_study, "level": left_level, "policy": left_policy,
            "reference_level": right_level, "reference_policy": right_policy,
            "contrast": label, "estimate": statistics.fmean(differences),
            "low": low, "high": high,
        })
    write_csv("paired_contrasts.csv", contrasts)
    generated = ROOT / "generated"
    lines = [
        f"\\newcommand{{\\ExtendedEpisodeCount}}{{{metadata['episodes']:,}}}",
        f"\\newcommand{{\\ExtendedCaseCount}}{{{metadata['cases']:,}}}",
        f"\\newcommand{{\\TotalEpisodeCount}}{{{metadata['episodes']+393216:,}}}",
    ]
    (generated / "extended_numbers.tex").write_text("\n".join(lines))
    for study in sorted({r["study"] for r in summaries}):
        selected = [r for r in summaries if r["study"] == study]
        lines = [r"\begin{longtable}{llrrrr}", r"\toprule",
                 r"Condition & Controller & Success (\%) & Low & High & Calls\\",
                 r"\midrule", r"\endhead"]
        for row in selected:
            lines.append(
                f"{escaped(row['level'])} & {LABELS[row['policy']]} & "
                f"{100*row['success']:.2f} & {100*row['low']:.2f} & "
                f"{100*row['high']:.2f} & {row['calls']:.2f}" + r"\\")
        lines += [r"\bottomrule", r"\end{longtable}"]
        (generated / f"ablation_{study}.tex").write_text("\n".join(lines))
    focus = [
        ("work", "n16_p0", "cache2"),
        ("memory", "p0", "adaptive2"),
        ("observation", "p0_shock1", "exact"),
        ("observation", "p1_shock1", "exact"),
        ("recovery", "r8_p0", "uniform_eliminate"),
        ("recovery", "r56_p0", "uniform_eliminate"),
        ("locality", "clustered_static", "adaptive2"),
        ("selection", "b8", "scan1_padded4"),
        ("selection", "b8", "scan4_fixed"),
        ("selection", "b8", "indexed"),
    ]
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Control & Controller & Success (\%) & Calls\\", r"\midrule"]
    for key in focus:
        row = next(r for r in summaries if (r["study"], r["level"], r["policy"]) == key)
        lines.append(f"{escaped(key[0]+': '+key[1])} & {LABELS[key[2]]} & "
                     f"{100*row['success']:.2f} & {row['calls']:.2f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "ablation_focus.tex").write_text("\n".join(lines))
    lines = [r"\begin{tabular}{rlrrrr}", r"\toprule",
             r"$b$ & Controller & Success (\%) & Checks/call & Build entries & Lookups\\",
             r"\midrule"]
    for row in summaries:
        if row["study"] != "selection" or row["level"] == "b2" or row["policy"] not in (
                "scan1", "scan1_padded4", "scan4_fixed", "indexed", "exact"):
            continue
        lines.append(
            f"{row['level'][1:]} & {LABELS[row['policy']]} & "
            f"{100*row['success']:.2f} & {row['evaluations']/row['calls']:.2f} & "
            f"{row['preparation_units']:.0f} & {row['lookups']:.2f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (generated / "selection_resources.tex").write_text("\n".join(lines))
    macros = []
    for study, level, policy, name in (
            ("memory", "p4", "adaptive2", "AdaptiveRefreshSuccess"),
            ("recovery", "r8_p0", "cache2_eliminate", "CacheRecoverySuccess"),
            ("recovery", "r56_p0", "uniform_eliminate", "EliminateCalls"),
            ("locality", "clustered_additive", "adaptive2", "ClusterSuccess")):
        row = next(r for r in summaries if (r["study"], r["level"], r["policy"]) == (
            study, level, policy))
        value = row["calls"] if name == "EliminateCalls" else 100*row["success"]
        macros.append(f"\\newcommand{{\\{name}}}{{{value:.2f}}}")
    with (generated / "extended_numbers.tex").open("a") as stream:
        stream.write("\n" + "\n".join(macros))
    interactions = full_interactions(samples, rng)
    report = {
        "analysis_sha256": source_hash, "summary_rows": len(summaries),
        "paired_contrast_rows": len(contrasts), "interaction_rows": len(interactions),
        "summaries": summaries, "contrasts": contrasts,
        "scope": protocol["claim_scope"],
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary_rows": len(summaries),
                      "contrast_rows": len(contrasts),
                      "interaction_rows": len(interactions),
                      "focus": [r for r in summaries if
                                (r["study"], r["level"], r["policy"]) in focus]},
                     indent=2))


def full_interactions(samples, rng):
    metadata = json.loads((ROOT / "results" / "metadata.json").read_text())
    if hashlib.sha256((ROOT / "results" / "episodes.csv").read_bytes()).hexdigest() != (
            metadata["episodes_sha256"]):
        raise RuntimeError("Original evidence changed since the primary run")
    names = ("depth", "branching", "width", "observation")
    endpoints = ((1, 8), (2, 8), (1, 8), ("snapshot", "history"))
    grouped = defaultdict(lambda: defaultdict(dict))
    with (ROOT / "results" / "episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            values = (int(row["depth"]), int(row["branching"]),
                      int(row["width"]), row["observation"])
            cells = grouped[(row["family"], row["policy"])][int(row["seed"])]
            if values in cells:
                raise RuntimeError("Duplicate original factorial cell")
            cells[values] = int(row["success"])
    rows = []
    for (family, policy), cells in sorted(grouped.items()):
        by_seed = {}
        for seed in range(256):
            seed_cells = cells[seed]
            endpoint_means = {}
            for endpoint_bits in product((0, 1), repeat=4):
                matches = [outcome for values, outcome in seed_cells.items()
                           if all(values[d] == endpoints[d][bit]
                                  for d, bit in enumerate(endpoint_bits))]
                if not matches:
                    raise RuntimeError("Missing factorial endpoint combination")
                endpoint_means[endpoint_bits] = statistics.fmean(matches)
            by_seed[seed] = endpoint_means
        for order in (2, 3, 4):
            for dims in combinations(range(4), order):
                changes = []
                for seed in range(256):
                    difference = 0.0
                    for signs in product((0, 1), repeat=order):
                        endpoint_bits = [None] * 4
                        for d, bit in zip(dims, signs):
                            endpoint_bits[d] = bit
                        sign = (-1)**(order-sum(signs))
                        other_dims = [d for d in range(4) if d not in dims]
                        means = [by_seed[seed][tuple(
                            endpoint_bits[d] if d in dims else filler[d]
                            for d in range(4))]
                            for free in product((0, 1), repeat=len(other_dims))
                            for filler in [{**{d: None for d in range(4)},
                                            **dict(zip(other_dims, free))}]]
                        difference += sign * statistics.fmean(means)
                    changes.append(difference)
                low, high = band(changes, samples, rng)
                rows.append({"family": family, "policy": policy, "order": order,
                             "interaction": "_x_".join(names[d] for d in dims),
                             "estimate": statistics.fmean(changes),
                             "low": low, "high": high})
    write_csv("all_endpoint_interactions.csv", rows)
    for order, filename in ((2, "pairwise"), (3, "higher")):
        selected = [r for r in rows if r["order"] == 2] if order == 2 else [
            r for r in rows if r["order"] > 2]
        columns = list(dict.fromkeys(r["interaction"] for r in selected))
        abbreviations = {
            "depth": "D", "branching": "B", "width": "W", "observation": "O"}
        headings = [r"$" + r"\times{} ".join(
            abbreviations[v] for v in name.split("_x_")) + "$" for name in columns]
        lines = [r"\begin{tabular}{ll" + "r"*len(columns) + "}", r"\toprule",
                 "Family & Controller & " + " & ".join(headings) + r"\\", r"\midrule"]
        for family, policy in sorted(grouped):
            values = [next(r for r in selected if r["family"] == family
                           and r["policy"] == policy and r["interaction"] == name)
                      for name in columns]
            lines.append(f"{family.capitalize()} & {escaped(policy)} & " +
                         " & ".join(f"{100*r['estimate']:.2f}" for r in values) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (ROOT / "generated" / f"{filename}_interactions.tex").write_text("\n".join(lines))
    return rows


if __name__ == "__main__":
    summarize()
