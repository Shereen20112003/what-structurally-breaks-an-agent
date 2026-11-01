"""Run the separately frozen follow-on design without rewriting primary data."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

from benchmark import structural_metrics
from extended_benchmark import cases, extended_rollout, predicted_success, prepare

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results_extended"


def run():
    protocol = json.loads((ROOT / "extended_protocol.json").read_text())
    design = list(cases())
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "case_design.json").write_text(json.dumps(
        [{"case_id": i, **asdict(case)} for i, case in enumerate(design)], indent=2))
    started = time.perf_counter()
    graph_cache, audits, count = {}, [], 0
    fields = [
        "case_id", "study", "level", "family", "depth", "branching", "width", "n",
        "period", "shock", "retries", "queries", "updates", "reverse", "policy",
        "seed", "success", "completed", "calls", "errors", "evaluations",
        "lookups", "preparation_units", "max_cache_entries", "max_cache_bits",
        "plan_slots", "snapshot_uses", "dependency_errors", "candidate_errors",
        "predicted", "prediction_kind",
    ]
    with (OUTPUT / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case_id, case in enumerate(design):
            c, v, p = case.config, case.variant, case.policy
            base = {
                "case_id": case_id, "study": case.study, "level": case.level,
                "family": c.family, "depth": c.depth, "branching": c.branching,
                "width": c.width, "n": c.n, "period": v.snapshot_every,
                "shock": int(v.hidden_resampling), "retries": v.retries,
                "queries": v.queries, "updates": v.updates,
                "reverse": int(v.reverse_labels), "policy": p.name,
            }
            graph_key = c.family, c.depth, c.n
            task = prepare(c, protocol["seed_start"], v)
            if graph_key not in graph_cache:
                graph_cache[graph_key] = structural_metrics(task)
            metrics = graph_cache[graph_key]
            audits.append({
                **base, "critical_path": metrics["critical_path"],
                "reachable_completion_masks": metrics["reachable_completion_masks"],
                "linear_extensions": metrics["linear_extensions"],
                "active_query_registers": len({node.query for node in task.nodes}),
                "edge_count": metrics["edges"], "optimal_calls": c.n,
                "mean_ready_nodes": metrics["mean_ready_nodes"],
            })
            for seed in range(protocol["seed_start"],
                              protocol["seed_start"]+protocol["seeds"]):
                task = prepare(c, seed, v)
                result = extended_rollout(task, v, p, seed*1009+7193)
                prediction = predicted_success(task, v, p)
                if prediction is not None and not 0 <= prediction <= 1+1e-12:
                    raise RuntimeError("Invalid analytic probability")
                positive = (
                    not v.hidden_resampling and p.scheduler == "ready" and
                    not p.uniform and
                    (p.memory_mode == "plan" or
                     (p.memory_mode != "stale" and p.scan >= c.branching and
                      (p.memory >= c.width or v.snapshot_every == 1))))
                positive |= (v.hidden_resampling and v.snapshot_every == 1
                             and p.name == "exact")
                positive |= (p.eliminate and not v.hidden_resampling and
                             v.retries >= c.n*(c.branching-1))
                if positive and not result["success"]:
                    raise RuntimeError(f"Positive control failed: {case}, seed={seed}")
                if result["calls"] > c.n+v.retries:
                    raise RuntimeError("Execution horizon violated")
                if p.memory_mode == "adaptive" and (
                        result["max_cache_entries"] > p.memory):
                    raise RuntimeError("Adaptive cache capacity exceeded")
                prediction_kind = (
                    "not_derived" if prediction is None else
                    "ensemble" if v.hidden_resampling and p.scan < c.branching else
                    "conditional_or_exact_population")
                writer.writerow({
                    **base, "seed": seed, **result,
                    "predicted": "" if prediction is None else prediction,
                    "prediction_kind": prediction_kind,
                })
                count += 1
            if case_id % 64 == 0:
                print(f"Completed {case_id+1}/{len(design)} cases", flush=True)
    with (OUTPUT / "structural_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)
    source_names = ["benchmark.py", "extended_benchmark.py", "run_extended.py",
                    "extended_protocol.json"]
    metadata = {
        "utc_finished": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter()-started,
        "cases": len(design), "episodes": count, "seeds": protocol["seeds"],
        "seed_start": protocol["seed_start"],
        "case_counts": dict(Counter(c.study for c in design)),
        "source_sha256": {name: hashlib.sha256(
            (ROOT / name).read_bytes()).hexdigest() for name in source_names},
        "episodes_sha256": hashlib.sha256(
            (OUTPUT / "episodes.csv").read_bytes()).hexdigest(),
        "model_calls": 0,
        "scope": protocol["claim_scope"],
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
