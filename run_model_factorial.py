"""Run a resumable, credit-metered fractional-factorial model study."""

from itertools import product
import json
from pathlib import Path
import random

from benchmark import Config, Environment, generate
from run_model_pilot import call_model, parse_action, prompt_for

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "model_factorial_protocol.json"
OUTPUT = ROOT / "results_model_factorial"
PRIOR_USAGE = ROOT / "results_model_pilot" / "usage"
SMOKE_USAGE = ROOT / "results" / "pilot_auto_smoke_usage.json"
NANO_PER_CREDIT = 1_000_000_000
STOP_AT_AI_CREDITS = 27.0
PRE_CALL_STOP_AT_AI_CREDITS = 26.75
PER_CALL_SOFT_CAP = 30
EXPECTED_MODEL = "gpt-6-luna"


def factorial_conditions(order_seed=20261101):
    conditions = []
    for depth_code, branching_code, width_code in product((-1, 1), repeat=3):
        observation_code = depth_code * branching_code * width_code
        codes = {
            "depth": depth_code,
            "branching": branching_code,
            "width": width_code,
            "observation": observation_code,
        }
        config = Config(
            "chains",
            2 if depth_code == -1 else 8,
            2 if branching_code == -1 else 8,
            1 if width_code == -1 else 8,
            "history" if observation_code == -1 else "snapshot",
        )
        condition_id = "_".join(
            f"{name[0]}{codes[name]:+d}"
            for name in ("depth", "branching", "width", "observation")
        )
        conditions.append({
            "condition_id": condition_id,
            "codes": codes,
            "config": config,
        })
    order = list(range(len(conditions)))
    random.Random(order_seed).shuffle(order)
    return [conditions[index] for index in order]


def prior_usage_nano_aiu():
    smoke = json.loads(SMOKE_USAGE.read_text(encoding="utf-8"))
    total = smoke["totalNanoAiu"]
    for path in sorted(PRIOR_USAGE.glob("call_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if type(report.get("totalNanoAiu")) is not int:
            raise RuntimeError(f"Usage report lacks totalNanoAiu: {path}")
        total += report["totalNanoAiu"]
    return total


def read_jsonl(path):
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        stream.flush()


def replay_attempt(task, attempt):
    env = Environment(task)
    observation = env.observe()
    history = []
    for turn in attempt.get("turns", []):
        if turn.get("stop_reason"):
            return env, observation, history, turn["stop_reason"]
        if turn.get("parse_error"):
            history.append(turn)
            observation = {
                **observation,
                "terminal": True,
                "success": False,
                "reason": turn["parse_error"],
            }
            return env, observation, history, None
        try:
            observation = env.execute(*turn["action"])
        except ValueError:
            return env, observation, history, "invalid_action"
        history.append(turn)
        if observation["terminal"]:
            break
    return env, observation, history, None


def run():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"].startswith("Prospective") is False:
        raise RuntimeError("Protocol must remain prospective before inference")
    budget = protocol["budget"]
    if (budget["total_project_stop_threshold_ai_credits"]
            != STOP_AT_AI_CREDITS
            or budget["pre_call_stop_threshold_ai_credits"]
            != PRE_CALL_STOP_AT_AI_CREDITS
            or budget["per_invocation_soft_cap_ai_credits"]
            != PER_CALL_SOFT_CAP):
        raise RuntimeError("Runner budget does not match frozen protocol")

    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "usage").mkdir(exist_ok=True)
    episodes_path = OUTPUT / "episodes.jsonl"
    attempts_path = OUTPUT / "attempts.jsonl"
    episodes = read_jsonl(episodes_path)
    attempts = read_jsonl(attempts_path)
    completed_keys = {
        (row["condition_id"], row["seed"]) for row in episodes
        if row.get("status") == "complete"
    }
    if len(completed_keys) != len(episodes):
        raise RuntimeError("Completed episode file contains duplicates or partial rows")
    latest_attempts = {}
    for attempt in attempts:
        latest_attempts[(attempt["condition_id"], attempt["seed"])] = attempt

    prior_nano = prior_usage_nano_aiu()
    usage_files = sorted((OUTPUT / "usage").glob("call_*.json"))
    new_nano = 0
    for path in usage_files:
        report = json.loads(path.read_text(encoding="utf-8"))
        if type(report.get("totalNanoAiu")) is not int:
            raise RuntimeError(f"Usage report lacks totalNanoAiu: {path}")
        new_nano += report["totalNanoAiu"]
    total_nano = prior_nano + new_nano
    if total_nano / NANO_PER_CREDIT >= PRE_CALL_STOP_AT_AI_CREDITS:
        print(f"Budget stop already reached: {total_nano / NANO_PER_CREDIT:.6f} AI credits")
        return

    usage_index = len(usage_files)
    for seed in protocol["seeds"]:
        for condition in factorial_conditions(protocol["condition_order_seed"]):
            condition_id = condition["condition_id"]
            key = (condition_id, seed)
            if key in completed_keys:
                continue
            config = condition["config"]
            task = generate(config, seed)
            previous = latest_attempts.get(key)
            env, observation, history, replay_stop = (
                replay_attempt(task, previous) if previous else
                (Environment(task), None, [], None)
            )
            if observation is None:
                observation = env.observe()
            model_ids_seen = (
                list(previous.get("model_ids_seen", [])) if previous else []
            )
            stop_reason = replay_stop

            while not stop_reason and not observation["terminal"]:
                total_credits = total_nano / NANO_PER_CREDIT
                if total_credits >= PRE_CALL_STOP_AT_AI_CREDITS:
                    stop_reason = "cumulative_ai_credit_stop"
                    break

                usage_path = OUTPUT / "usage" / f"call_{usage_index:04d}.json"
                raw, usage, model_ids = call_model(
                    prompt_for(task, observation, history),
                    usage_path,
                    max_ai_credits=PER_CALL_SOFT_CAP,
                )
                usage_index += 1
                total_nano += usage["totalNanoAiu"]
                model_ids_seen.extend(model_ids)
                if model_ids != [EXPECTED_MODEL]:
                    turn = {
                        "observation": observation,
                        "raw_response": raw,
                        "action": None,
                        "parse_error": None,
                        "stop_reason": "unexpected_model_route",
                        "usage_file": usage_path.relative_to(ROOT).as_posix(),
                        "nano_aiu": usage["totalNanoAiu"],
                        "premium_request_cost":
                            usage.get("totalPremiumRequestCost"),
                        "model_ids": model_ids,
                    }
                    history.append(turn)
                    stop_reason = "unexpected_model_route"
                else:
                    action, parse_error = parse_action(raw)
                    turn = {
                        "observation": observation,
                        "raw_response": raw,
                        "action": list(action) if action else None,
                        "parse_error": parse_error,
                        "usage_file": usage_path.relative_to(ROOT).as_posix(),
                        "nano_aiu": usage["totalNanoAiu"],
                        "premium_request_cost":
                            usage.get("totalPremiumRequestCost"),
                        "model_ids": model_ids,
                    }
                    if parse_error:
                        turn["result"] = {
                            "terminal": True,
                            "success": False,
                            "reason": parse_error,
                        }
                        stop_reason = parse_error
                        observation = {
                            **observation,
                            "terminal": True,
                            "success": False,
                            "reason": parse_error,
                        }
                    else:
                        try:
                            following = env.execute(*action)
                        except ValueError:
                            following = env.observe()
                            following["terminal"] = True
                            following["reason"] = "invalid_action"
                            stop_reason = "invalid_action"
                        turn["result"] = following
                        observation = following
                    history.append(turn)

                checkpoint = {
                    "condition_id": condition_id,
                    "codes": condition["codes"],
                    "seed": seed,
                    "config": config.__dict__,
                    "status": "in_progress",
                    "turns": history,
                    "model_ids_seen": model_ids_seen,
                    "stop_reason": stop_reason,
                    "cumulative_nano_aiu": total_nano,
                }
                append_jsonl(attempts_path, checkpoint)
                latest_attempts[key] = checkpoint
                if (total_nano / NANO_PER_CREDIT >= STOP_AT_AI_CREDITS
                        and not observation["terminal"]):
                    stop_reason = "cumulative_ai_credit_stop"

            if stop_reason in ("cumulative_ai_credit_stop",
                               "unexpected_model_route"):
                print(
                    f"Stopped before completing {condition_id}, seed {seed}: "
                    f"{stop_reason}; cumulative="
                    f"{total_nano / NANO_PER_CREDIT:.6f} AI credits"
                )
                return

            if not observation["terminal"]:
                raise RuntimeError("Episode ended without a terminal state")
            episode = {
                "condition_id": condition_id,
                "codes": condition["codes"],
                "seed": seed,
                "config": config.__dict__,
                "status": "complete",
                "success": int(observation["success"]),
                "completed": len(observation["done"]),
                "decisions": len(history),
                "failure": observation["reason"] or (
                    history[-1].get("parse_error") if history else None
                ) or "none",
                "model_ids_seen": model_ids_seen,
                "turns": history,
                "cumulative_nano_aiu": total_nano,
            }
            append_jsonl(episodes_path, episode)
            completed_keys.add(key)
            print(
                f"{len(completed_keys)}/16 {condition_id} seed={seed} "
                f"success={episode['success']} completed={episode['completed']} "
                f"cumulative_aiu={total_nano / NANO_PER_CREDIT:.3f}"
            )
    print(
        f"Complete: {len(completed_keys)}/16 episodes; cumulative="
        f"{total_nano / NANO_PER_CREDIT:.6f} AI credits"
    )


if __name__ == "__main__":
    run()
