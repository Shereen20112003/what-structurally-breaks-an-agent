"""Execute the frozen local design and persist every rollout and audit."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

from benchmark import (POLICIES, generate, grid, rollout, structural_metrics,
                       task_expected_success)


ROOT = Path(__file__).resolve().parent


def run(seeds):
    started = time.perf_counter()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if seeds != protocol["seeds"]:
        raise ValueError("Primary seed count must match the frozen protocol")
    fields = ["family", "depth", "branching", "width", "observation", "seed",
              "policy", "success", "completed", "calls", "evaluations", "failure",
              "expected_success"]
    audit_fields = None
    audits = []
    count = 0
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for config in grid():
            audit = {**asdict(config),
                     **structural_metrics(generate(config, 0))}
            audits.append(audit)
            audit_fields = list(audit)
            for seed in range(seeds):
                task = generate(config, seed)
                for index, policy in enumerate(POLICIES):
                    result = rollout(task, policy, seed * 1009 + index * 9176)
                    if policy.name == "exact" and not result["success"]:
                        raise RuntimeError(f"Solvability failure: {config}, {seed}")
                    writer.writerow({
                        **{k: v for k, v in asdict(config).items() if k != "n"},
                        "seed": seed, "policy": policy.name, **result,
                        "expected_success": task_expected_success(task, policy),
                    })
                    count += 1
            print(f"Completed {config}", flush=True)
    with (output / "structural_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audits)
    example = generate(next(c for c in grid()
                            if c.family == "chains" and c.depth == 4 and
                            c.branching == 4 and c.width == 4 and
                            c.observation == "history"), 17)
    (output / "example_trace.json").write_text(json.dumps(
        {"task": asdict(example),
         "episode": rollout(example, POLICIES[0], 17, trace=True)}, indent=2))
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ("benchmark.py", "run_experiments.py", "protocol.json")}
    metadata = {
        "utc_finished": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "seeds": seeds, "tasks": len(audits) * seeds, "episodes": count,
        "seconds": time.perf_counter() - started, "source_sha256": hashes,
        "episodes_sha256": hashlib.sha256(
            (output / "episodes.csv").read_bytes()).hexdigest(),
        "model_calls": 0, "scope": protocol["scope"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=256)
    run(parser.parse_args().seeds)
