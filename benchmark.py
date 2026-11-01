"""Finite, stateful workflows with observation-only reference controllers."""

from dataclasses import dataclass
from itertools import product
from math import factorial
import random


@dataclass(frozen=True)
class Config:
    family: str
    depth: int
    branching: int
    width: int
    observation: str
    n: int = 8

    def __post_init__(self):
        if self.family not in ("chains", "barriers"):
            raise ValueError("Unknown graph family")
        if self.n < 1 or self.depth < 1 or self.n % self.depth:
            raise ValueError("Depth must divide the operation count")
        if self.branching not in (2, 4, 8):
            raise ValueError("Branching must be 2, 4, or 8")
        if self.width not in (1, 2, 4, 8) or self.n % self.width:
            raise ValueError("Width must divide the operation count")
        if self.observation not in ("snapshot", "history"):
            raise ValueError("Unknown observation channel")


def grid():
    for values in product(("chains", "barriers"), (1, 2, 4, 8),
                          (2, 4, 8), (1, 2, 4, 8),
                          ("snapshot", "history")):
        yield Config(*values)


@dataclass(frozen=True)
class Node:
    prerequisites: tuple[int, ...]
    query: int
    update: int
    delta: int
    options: tuple[int, ...]


@dataclass(frozen=True)
class Task:
    config: Config
    seed: int
    initial: tuple[int, ...]
    nodes: tuple[Node, ...]


def generate(config, seed):
    rng = random.Random(seed)
    # Draw fixed-size randomness before selecting any structural level.
    initial = tuple(rng.randrange(8) for _ in range(8))
    labels = list(range(config.n))
    rng.shuffle(labels)
    queries = list(range(config.n))
    rng.shuffle(queries)
    updates = [rng.randrange(8) for _ in range(config.n)]
    deltas = [rng.randrange(1, 8) for _ in range(config.n)]
    permutations = []
    for _ in range(config.n):
        slots = list(range(8))
        rng.shuffle(slots)
        permutations.append(slots)
    per_layer = config.n // config.depth
    nodes = [None] * config.n
    for position, label in enumerate(labels):
        layer, column = divmod(position, per_layer)
        if not layer:
            prerequisites = ()
        elif config.family == "chains":
            prerequisites = (labels[position - per_layer],)
        else:
            prerequisites = tuple(labels[(layer - 1) * per_layer:
                                          layer * per_layer])
        nodes[label] = Node(
            prerequisites, queries[position] % config.width,
            updates[position] % config.width, deltas[position],
            tuple(x for x in permutations[position] if x < config.branching))
    return Task(config, seed, initial, tuple(nodes))


class Environment:
    def __init__(self, task):
        self.task = task
        self._state = list(task.initial)
        self._done = set()
        self._terminal = False
        self.calls = 0

    def observe(self):
        snapshot = (not self.calls or
                    self.task.config.observation == "snapshot")
        return {
            "state": tuple(self._state) if snapshot else (None,) * 8,
            "done": tuple(sorted(self._done)),
            "terminal": self._terminal,
            "success": len(self._done) == self.task.config.n,
            "update": None,
            "reason": None,
        }

    def execute(self, node_id, option):
        if self._terminal:
            raise RuntimeError("Cannot act after termination")
        if type(node_id) is not int or not 0 <= node_id < len(self.task.nodes):
            raise ValueError("Invalid node identifier")
        node = self.task.nodes[node_id]
        if type(option) is not int or not 0 <= option < len(node.options):
            raise ValueError("Invalid candidate slot")
        self.calls += 1
        reason = None
        if node_id in self._done or not set(node.prerequisites) <= self._done:
            reason = "dependency"
        elif node.options[option] != (
                self._state[node.query] % self.task.config.branching):
            reason = "candidate"
        if reason:
            self._terminal = True
        else:
            self._done.add(node_id)
            self._state[node.update] = (
                self._state[node.update] + node.delta) % 8
            self._terminal = len(self._done) == self.task.config.n
        result = self.observe()
        result["reason"] = reason
        if not reason:
            result["update"] = (node.update, node.delta)
        return result


@dataclass(frozen=True)
class Policy:
    name: str
    memory: int = 8
    scan: int = 8
    scheduler: str = "ready"
    uniform: bool = False


POLICIES = (
    Policy("exact"),
    Policy("memory2", memory=2),
    Policy("memory4", memory=4),
    Policy("scan1", scan=1),
    Policy("scan4", scan=4),
    Policy("memory2_scan4", memory=2, scan=4),
    Policy("random_order", scheduler="random"),
    Policy("uniform", uniform=True),
)


class Controller:
    """Consumes public task rules and observations, never environment state."""

    def __init__(self, task, policy, seed):
        self.task, self.policy = task, policy
        self.rng = random.Random(seed)
        self.memory = {}
        self.initialized = False
        self.evaluations = 0

    def act(self, observation):
        if not self.initialized:
            self.memory = {i: observation["state"][i]
                           for i in range(min(self.policy.memory,
                                              self.task.config.width))}
            self.initialized = True
        elif observation["update"] is not None:
            register, delta = observation["update"]
            if register in self.memory:
                self.memory[register] = (self.memory[register] + delta) % 8
        done = set(observation["done"])
        candidates = [i for i, node in enumerate(self.task.nodes)
                      if i not in done and
                      (self.policy.scheduler == "random" or
                       set(node.prerequisites) <= done)]
        if not candidates:
            raise RuntimeError("No legal scheduling candidate")
        node_id = (self.rng.choice(candidates)
                   if self.policy.scheduler == "random" else min(candidates))
        node = self.task.nodes[node_id]
        value = observation["state"][node.query]
        if value is None:
            value = self.memory.get(node.query)
        if value is None or self.policy.uniform:
            return node_id, self.rng.randrange(self.task.config.branching)
        target = value % self.task.config.branching
        limit = min(self.policy.scan, len(node.options))
        for slot in range(limit):
            self.evaluations += 1
            if node.options[slot] == target:
                return node_id, slot
        return node_id, self.rng.randrange(limit, len(node.options))


def rollout(task, policy, seed, trace=False):
    env = Environment(task)
    controller = Controller(task, policy, seed)
    observation = env.observe()
    records = []
    while not observation["terminal"]:
        action = controller.act(observation)
        following = env.execute(*action)
        if trace:
            records.append({"observation": observation, "action": action,
                            "result": following})
        observation = following
    return {
        "success": int(observation["success"]),
        "completed": len(observation["done"]),
        "calls": env.calls,
        "evaluations": controller.evaluations,
        "failure": observation["reason"] or "none",
        **({"trace": records} if trace else {}),
    }


def structural_metrics(task):
    """Enumerate reachable completion masks, not arbitrary register states."""
    masks = [sum(1 << p for p in node.prerequisites) for node in task.nodes]
    ways = {0: 1}
    ready_counts = []
    for mask in range(1 << task.config.n):
        if mask not in ways:
            continue
        ready = [i for i, prerequisites in enumerate(masks)
                 if not mask & (1 << i) and mask & prerequisites == prerequisites]
        if ready:
            ready_counts.append(len(ready))
        for node in ready:
            following = mask | (1 << node)
            ways[following] = ways.get(following, 0) + ways[mask]
    lengths = {}

    def depth(i):
        if i not in lengths:
            lengths[i] = 1 + max((depth(p) for p in
                                 task.nodes[i].prerequisites), default=0)
        return lengths[i]

    c = task.config
    return {
        "critical_path": max(depth(i) for i in range(c.n)),
        "operations": c.n,
        "optimal_calls": c.n,
        "candidate_count": c.branching,
        "active_registers": len({node.query for node in task.nodes}),
        "initial_state_entropy_bits": 3 * c.width,
        "snapshot_entropy_bits": 3 * c.width if c.observation == "snapshot" else 0,
        "edges": sum(len(node.prerequisites) for node in task.nodes),
        "reachable_completion_masks": len(ways),
        "linear_extensions": ways[(1 << c.n) - 1],
        "mean_ready_nodes": sum(ready_counts) / len(ready_counts),
        "mean_admissible_action_slots":
            c.branching * sum(ready_counts) / len(ready_counts),
    }


def expected_success(config, policy):
    b, n, w = config.branching, config.n, config.width
    if policy.uniform:
        return b ** (-n)
    if policy.scheduler == "random":
        if config.family == "chains":
            return factorial(config.depth) ** (-n // config.depth)
        return (factorial(n // config.depth) ** config.depth) / factorial(n)
    raise ValueError("This policy requires a task-specific prediction")


def task_expected_success(task, policy):
    if policy.uniform or policy.scheduler == "random":
        return expected_success(task.config, policy)
    remaining = set(range(task.config.n))
    done = set()
    state = list(task.initial)
    probability = 1.0
    while remaining:
        node_id = min(i for i in remaining
                      if set(task.nodes[i].prerequisites) <= done)
        node = task.nodes[node_id]
        visible = (not done or task.config.observation == "snapshot"
                   or node.query < policy.memory)
        if not visible:
            probability /= task.config.branching
        else:
            slot = node.options.index(state[node.query] % task.config.branching)
            limit = min(policy.scan, task.config.branching)
            if slot >= limit:
                probability /= task.config.branching - limit
        state[node.update] = (state[node.update] + node.delta) % 8
        remaining.remove(node_id)
        done.add(node_id)
    return probability
