"""Run a small, credit-metered interactive pilot through GitHub Copilot CLI."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from benchmark import Config, Environment, generate

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "model_pilot_protocol.json"
OUTPUT = ROOT / "results_model_pilot"
EXTENSION_PROTOCOL_PATH = ROOT / "model_pilot_extension_protocol.json"
STOP_CREDITS = 28.0
NANO_PER_CREDIT = 1_000_000_000
SMOKE_USAGE_PATH = ROOT / "results" / "pilot_auto_smoke_usage.json"


def intervention_configs():
    protocol = json.loads(EXTENSION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    for factor, pair in protocol["interventions"].items():
        for level, settings in pair.items():
            yield factor, level, Config("chains", **settings)


def parse_action(text):
    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if (not isinstance(action, dict) or set(action) != {"node_id", "slot"}
            or type(action["node_id"]) is not int
            or type(action["slot"]) is not int):
        return None, "invalid_action_schema"
    return (action["node_id"], action["slot"]), None


def prompt_for(task, observation, history):
    nodes = [
        {
            "id": index,
            "prerequisites": list(node.prerequisites),
            "query_register": node.query,
            "update_register": node.update,
            "update_amount": node.delta,
            "candidate_codes_by_slot": list(node.options),
        }
        for index, node in enumerate(task.nodes)
    ]
    task_description = {
        "operation_count": task.config.n,
        "candidate_count": task.config.branching,
        "active_registers": task.config.width,
        "feedback": task.config.observation,
        "initial_registers": list(task.initial),
        "operations": nodes,
    }
    return (
        "You control a deterministic workflow. Choose exactly one operation and "
        "candidate slot for the next action. An operation is legal only when all "
        "its prerequisites are complete. Its correct slot is the index whose "
        "candidate code equals the queried register value modulo candidate_count. "
        "After success, add update_amount to update_register modulo 8. "
        "Return exactly one JSON object with integer keys node_id and slot; "
        "do not include prose or markdown.\n"
        f"TASK={json.dumps(task_description, separators=(',', ':'))}\n"
        f"OBSERVATION={json.dumps(observation, separators=(',', ':'))}\n"
        f"INTERACTION_HISTORY={json.dumps(history, separators=(',', ':'))}\n"
        "Return the next action now."
    )


def call_model(prompt, usage_path, max_ai_credits=30):
    command = [
        "copilot", "--model", "auto", "--max-ai-credits",
        str(max_ai_credits),
        "--usage-output-file", str(usage_path), "--prompt", prompt, "--silent",
    ]
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(
            f"Copilot invocation failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if not usage_path.is_file():
        raise RuntimeError(f"Copilot usage report was not created: {usage_path}")
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    if type(usage.get("totalNanoAiu")) is not int:
        raise RuntimeError(f"Usage report lacks totalNanoAiu: {usage_path}")
    model_ids = list(usage.get("modelMetrics", {}))
    return result.stdout.strip(), usage, model_ids


def run_episode(factor, level, task, seed, total_nano_aiu, usage_index):
    env = Environment(task)
    observation = env.observe()
    history = []
    model_ids_seen = []
    stop_reason = None

    while not observation["terminal"]:
        if total_nano_aiu / NANO_PER_CREDIT >= STOP_CREDITS:
            stop_reason = "credit_threshold"
            break
        usage_path = OUTPUT / "usage" / f"call_{usage_index:04d}.json"
        prompt = prompt_for(task, observation, history)
        raw, usage, model_ids = call_model(prompt, usage_path)
        total_nano_aiu += usage["totalNanoAiu"]
        usage_index += 1
        model_ids_seen.extend(model_ids)
        action, parse_error = parse_action(raw)
        turn = {
            "observation": observation,
            "raw_response": raw,
            "action": action,
            "parse_error": parse_error,
            "usage_file": usage_path.relative_to(ROOT).as_posix(),
            "nano_aiu": usage["totalNanoAiu"],
            "premium_request_cost": usage.get("totalPremiumRequestCost"),
            "model_ids": model_ids,
        }
        if parse_error:
            turn["result"] = {"terminal": True, "success": False,
                              "reason": parse_error}
            history.append(turn)
            stop_reason = parse_error
            break
        try:
            following = env.execute(*action)
        except ValueError:
            turn["result"] = {"terminal": True, "success": False,
                              "reason": "invalid_action"}
            history.append(turn)
            stop_reason = "invalid_action"
            break
        turn["result"] = following
        history.append(turn)
        observation = following

    episode = {
        "factor": factor,
        "level": level,
        "seed": seed,
        "config": asdict(task.config),
        "success": int(observation["success"]) if stop_reason is None else 0,
        "completed": len(observation["done"]),
        "decisions": len(history),
        "failure": stop_reason or observation["reason"] or "none",
        "model_ids_seen": model_ids_seen,
        "turns": history,
        "cumulative_nano_aiu": total_nano_aiu,
    }
    return episode, total_nano_aiu, usage_index


def run():
    if not SMOKE_USAGE_PATH.is_file():
        raise FileNotFoundError(f"Budget baseline is missing: {SMOKE_USAGE_PATH}")
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "usage").mkdir(exist_ok=True)
    protocol = json.loads(EXTENSION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    smoke_usage = json.loads(SMOKE_USAGE_PATH.read_text(encoding="utf-8"))
    previous_path = OUTPUT / "summary.json"
    previous = (json.loads(previous_path.read_text(encoding="utf-8"))
                if previous_path.is_file() else None)
    if (OUTPUT / "episodes.jsonl").is_file():
        recovered = [
            json.loads(line)
            for line in (OUTPUT / "episodes.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    else:
        recovered = []
    if recovered and (previous is None or
                      len(recovered) > len(previous.get("episodes", []))):
        counts = {}
        for episode in recovered:
            key = (episode["factor"], episode["level"], episode["seed"])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > 1:
                episode["block"] = "recovery_duplicate_unplanned"
        usage_reports = sorted((OUTPUT / "usage").glob("call_*.json"))
        total_nano = smoke_usage["totalNanoAiu"]
        for usage_report in usage_reports:
            total_nano += json.loads(
                usage_report.read_text(encoding="utf-8")
            )["totalNanoAiu"]
        previous = {
            "protocol": json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")),
            "episodes": recovered,
            "total_nano_aiu_including_smoke": total_nano,
        }
    if previous:
        completed = previous["episodes"]
        total_nano_aiu = previous["total_nano_aiu_including_smoke"]
        usage_index = sum(len(episode["turns"]) for episode in completed)
        if total_nano_aiu / NANO_PER_CREDIT >= STOP_CREDITS:
            print("Credit stop threshold already reached; no further calls.")
            return
    else:
        total_nano_aiu = smoke_usage["totalNanoAiu"]
        usage_index = 0
        completed = []
    completed_keys = {
        (episode["factor"], episode["level"], episode["seed"])
        for episode in completed
    }
    extension_episodes = []
    for factor, level, config in intervention_configs():
        seed = protocol["paired_seed"]
        if (factor, level, seed) in completed_keys:
            continue
        if total_nano_aiu / NANO_PER_CREDIT >= STOP_CREDITS:
            break
        task = generate(config, seed)
        episode, total_nano_aiu, usage_index = run_episode(
            factor, level, task, seed, total_nano_aiu, usage_index
        )
        episode["block"] = "paired_seed_20262_extension"
        completed.append(episode)
        extension_episodes.append(episode)
        with (OUTPUT / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(episode) + "\n")
        print(
            f"{factor}/{level} seed={seed}: success={episode['success']} "
            f"completed={episode['completed']} credits="
            f"{total_nano_aiu / NANO_PER_CREDIT:.3f}",
            flush=True,
        )

    summary = {
        "status": "exploratory_pilot",
        "protocol": previous["protocol"] if previous else json.loads(
            PROTOCOL_PATH.read_text(encoding="utf-8")),
        "extension_protocol": protocol,
        "episodes_completed": len(completed),
        "episodes": completed,
        "extension_episodes_completed": sum(
            episode.get("block") == "paired_seed_20262_extension"
            for episode in completed
        ),
        "total_nano_aiu_including_smoke": total_nano_aiu,
        "total_ai_credits_including_smoke":
            total_nano_aiu / NANO_PER_CREDIT,
        "total_premium_request_cost_including_smoke":
            smoke_usage.get("totalPremiumRequestCost", 0) + sum(
                turn.get("premium_request_cost") or 0
                for episode in completed for turn in episode["turns"]
            ),
        "model_ids_seen": sorted({
            model_id for episode in completed
            for model_id in episode["model_ids_seen"]
        }),
        "stop_reason": (
            "credit_threshold"
            if total_nano_aiu / NANO_PER_CREDIT >= STOP_CREDITS
            else "protocol_complete"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "episodes"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
