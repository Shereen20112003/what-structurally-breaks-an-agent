"""Run the frozen, no-API local-model factorial study."""

from dataclasses import asdict
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from benchmark import Config, Environment, generate

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "local_model_protocol.json"
OUTPUT = ROOT / "results_local_model"
DEFAULT_MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_SERVER = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
RUNTIME_BUILD = "llama.cpp b12030 CPU x64"
RUNTIME_ARCHIVE_SHA256 = (
    "0a4ed4275235508d5e25d31e1a5586ae4e1a08079d09b3a1a78f476f788918c6"
)
FACTORS = {
    "family": ("chains", "barriers"),
    "depth": (2, 8),
    "branching": (2, 8),
    "width": (1, 8),
    "observation": ("history", "snapshot"),
}
SEEDS = (41000, 41001, 41002, 41003, 41004, 41005, 41006, 41007)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalModelClient:
    def __init__(self, model_path, server_path):
        self.model_path = Path(model_path)
        self.server_path = Path(server_path)
        self.process = None
        self.port = None
        self.version = None
        self.log_path = OUTPUT / "llama-server.log"

    def __enter__(self):
        if not self.server_path.is_file():
            raise FileNotFoundError(f"llama-server executable is missing: {self.server_path}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        version = subprocess.run(
            [str(self.server_path), "--version"],
            cwd=self.server_path.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        self.version = (version.stdout or version.stderr).strip()
        command = [
            str(self.server_path),
            "--model", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", "8192",
            "--threads", "4",
            "--threads-batch", "4",
            "--n-gpu-layers", "0",
            "--no-webui",
        ]
        with self.log_path.open("ab") as log:
            self.process = subprocess.Popen(
                command,
                cwd=self.server_path.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            self._wait_until_ready()
        except Exception:
            self._stop()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop()

    def _stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    def _wait_until_ready(self):
        deadline = time.monotonic() + 300
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited during startup. "
                    f"See runtime log: {self.log_path}"
                )
            try:
                with urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, URLError):
                time.sleep(1)
        raise TimeoutError(
            f"llama-server did not become ready; see runtime log: {self.log_path}"
        )

    def create_chat_completion(
        self, messages, temperature=0, max_tokens=64, seed=None
    ):
        payload = json.dumps({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                return json.loads(response.read())
        except URLError as exc:
            raise RuntimeError(f"Local llama-server request failed: {exc}") from exc


class EpisodeExecutionError(RuntimeError):
    def __init__(self, message, partial_episode):
        super().__init__(message)
        self.partial_episode = partial_episode


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


def task_prompt(task, observation, history):
    spec = {
        "operation_count": task.config.n,
        "candidate_count": task.config.branching,
        "active_registers": task.config.width,
        "feedback": task.config.observation,
        "initial_registers": list(task.initial),
        "operations": [
            {
                "id": index,
                "prerequisites": list(node.prerequisites),
                "query_register": node.query,
                "update_register": node.update,
                "update_amount": node.delta,
                "candidate_codes_by_slot": list(node.options),
            }
            for index, node in enumerate(task.nodes)
        ],
    }
    return (
        "Control a deterministic workflow. Choose one unfinished operation "
        "whose prerequisites are complete. Its correct slot is the index "
        "whose listed candidate code equals the queried register value "
        "modulo candidate_count. On success, the update amount is added to "
        "the update register modulo 8. Return exactly JSON with integer "
        "fields node_id and slot; no explanation.\n"
        f"TASK={json.dumps(spec, separators=(',', ':'))}\n"
        f"OBSERVATION={json.dumps(observation, separators=(',', ':'))}\n"
        f"HISTORY={json.dumps(history, separators=(',', ':'))}\n"
        "Return the next action."
    )


def factorial_cases():
    keys = tuple(FACTORS)
    for values in itertools.product(*(FACTORS[key] for key in keys)):
        settings = dict(zip(keys, values, strict=True))
        family = settings.pop("family")
        yield family, Config(family=family, n=8, **settings)


def cell_key(episode):
    config = episode["config"]
    return (
        episode["seed"], config["family"], config["depth"],
        config["branching"], config["width"], config["observation"],
    )


def run_episode(model, family, config, seed):
    task = generate(config, seed)
    env = Environment(task)
    observation = env.observe()
    history = []
    decisions = []
    first_failure = None
    while not observation["terminal"]:
        prompt = task_prompt(task, observation, history)
        started = time.perf_counter()
        try:
            response = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Follow the task rules exactly."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=64,
                seed=seed,
            )
        except (Exception, KeyboardInterrupt) as exc:
            partial_episode = {
                "seed": seed,
                "factorial_cell": {
                    "family": family,
                    "depth": config.depth,
                    "branching": config.branching,
                    "width": config.width,
                    "observation": config.observation,
                },
                "config": asdict(config),
                "success": 0,
                "completed": len(observation["done"]),
                "decisions": decisions,
                "first_failure": "inference_error",
                "attempt_complete": False,
                "inference_error": f"{type(exc).__name__}: {exc}",
            }
            raise EpisodeExecutionError(
                f"Inference failed for seed {seed} in {family} "
                f"(depth={config.depth}, branching={config.branching}, "
                f"width={config.width}, observation={config.observation}).",
                partial_episode,
            ) from exc
        elapsed = time.perf_counter() - started
        text = response["choices"][0]["message"]["content"] or ""
        action, parse_error = parse_action(text)
        decision = {
            "response": text,
            "action": action,
            "parse_error": parse_error,
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "elapsed_seconds": elapsed,
        }
        if parse_error:
            first_failure = parse_error
            decisions.append(decision)
            break
        try:
            following = env.execute(*action)
        except ValueError:
            decision["result"] = {"terminal": True, "success": False,
                                  "reason": "invalid_action"}
            decisions.append(decision)
            first_failure = "invalid_action"
            break
        decision["result"] = following
        decisions.append(decision)
        history.append({
            "observation": observation,
            "action": action,
            "result": following,
        })
        observation = following
        if observation["reason"]:
            first_failure = observation["reason"]

    return {
        "seed": seed,
        "factorial_cell": {
            "family": family,
            "depth": config.depth,
            "branching": config.branching,
            "width": config.width,
            "observation": config.observation,
        },
        "config": asdict(config),
        "success": int(observation["success"]),
        "completed": len(observation["done"]),
        "decisions": decisions,
        "first_failure": first_failure or "none",
    }


def run(model_path=DEFAULT_MODEL, max_episodes=None, smoke=False,
        server_path=DEFAULT_SERVER):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Local model file is missing: {model_path}")
    server_path = Path(server_path)
    OUTPUT.mkdir(exist_ok=True)
    episodes_path = OUTPUT / "episodes.jsonl"
    existing = []
    if episodes_path.is_file():
        existing = [
            json.loads(line) for line in
            episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {cell_key(episode) for episode in existing}
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected_hash = sha256_file(model_path)
    if expected_hash != protocol["model"]["sha256"]:
        raise RuntimeError(
            "The local model file does not match the SHA-256 frozen in the protocol."
        )
    print(
        f"Model={protocol['model']['repository']} sha256={expected_hash}; "
        f"runtime={RUNTIME_BUILD}; local CPU inference only.",
        flush=True,
    )
    with LocalModelClient(model_path, server_path) as model:
        if smoke:
            family, config = next(factorial_cases())
            episode = run_episode(model, family, config, 51000)
            episode["smoke_test"] = True
            episode["model"] = protocol["model"]["repository"]
            episode["model_file_sha256"] = expected_hash
            episode["runtime"] = {
                "build": RUNTIME_BUILD,
                "version": model.version,
                "archive_sha256": RUNTIME_ARCHIVE_SHA256,
            }
            smoke_path = OUTPUT / "smoke.jsonl"
            with smoke_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(episode, separators=(",", ":")) + "\n")
            print(
                f"Smoke test (excluded from study): success={episode['success']}, "
                f"decisions={len(episode['decisions'])}; saved to {smoke_path}",
                flush=True,
            )
            return

        count = 0
        with episodes_path.open("a", encoding="utf-8") as stream:
            for family, config in factorial_cases():
                for seed in SEEDS:
                    probe = {
                        "seed": seed,
                        "config": asdict(config),
                    }
                    key = cell_key(probe)
                    if key in completed:
                        continue
                    if max_episodes is not None and count >= max_episodes:
                        return
                    try:
                        episode = run_episode(model, family, config, seed)
                    except EpisodeExecutionError as exc:
                        exc.partial_episode.update({
                            "model": protocol["model"]["repository"],
                            "model_file_sha256": expected_hash,
                            "runtime": {
                                "build": RUNTIME_BUILD,
                                "version": model.version,
                                "archive_sha256": RUNTIME_ARCHIVE_SHA256,
                            },
                        })
                        attempts_path = OUTPUT / "attempts.jsonl"
                        with attempts_path.open("a", encoding="utf-8") as attempts:
                            attempts.write(json.dumps(
                                exc.partial_episode, separators=(",", ":")
                            ) + "\n")
                            attempts.flush()
                        raise
                    episode["model"] = protocol["model"]["repository"]
                    episode["model_file_sha256"] = expected_hash
                    episode["attempt_complete"] = True
                    episode["runtime"] = {
                        "build": RUNTIME_BUILD,
                        "version": model.version,
                        "archive_sha256": RUNTIME_ARCHIVE_SHA256,
                    }
                    stream.write(json.dumps(episode, separators=(",", ":")) + "\n")
                    stream.flush()
                    count += 1
                    completed.add(key)
                    print(
                        f"{count}: {family}, d={config.depth}, "
                        f"b={config.branching}, w={config.width}, "
                        f"{config.observation}, seed={seed}, "
                        f"success={episode['success']}, "
                        f"decisions={len(episode['decisions'])}",
                        flush=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.model, args.max_episodes, args.smoke, args.server)
