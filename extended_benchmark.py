"""Follow-on controls; the original benchmark and its evidence stay unchanged."""

from collections import OrderedDict
from dataclasses import dataclass, replace
from math import comb
import random

from benchmark import Config, Task, generate


@dataclass(frozen=True)
class Variant:
    snapshot_every: int = 0
    hidden_resampling: bool = False
    retries: int = 0
    queries: str = "balanced"
    updates: str = "additive"
    reverse_labels: bool = False

    def __post_init__(self):
        if type(self.snapshot_every) is not int or self.snapshot_every < 0:
            raise ValueError("Snapshot period must be a nonnegative integer")
        if type(self.retries) is not int or self.retries < 0:
            raise ValueError("Retry allowance must be a nonnegative integer")
        if self.queries not in ("balanced", "clustered", "concentrated"):
            raise ValueError("Unknown query assignment")
        if self.updates not in ("additive", "static"):
            raise ValueError("Unknown update rule")


@dataclass(frozen=True)
class ExtendedPolicy:
    name: str
    memory: int = 8
    scan: int = 8
    memory_mode: str = "fixed"
    selection: str = "serial"
    scheduler: str = "ready"
    uniform: bool = False
    eliminate: bool = False

    def __post_init__(self):
        if not 0 <= self.memory <= 8 or not 0 <= self.scan <= 8:
            raise ValueError("Unsupported cache or scan capacity")
        if self.memory_mode not in ("fixed", "adaptive", "modulo", "plan", "stale"):
            raise ValueError("Unknown retention strategy")
        if self.selection not in ("serial", "fixed", "padded", "indexed"):
            raise ValueError("Unknown selection strategy")
        if self.scheduler not in ("ready", "random"):
            raise ValueError("Unknown scheduler")


EXACT = ExtendedPolicy("exact")
CACHE2 = ExtendedPolicy("cache2", memory=2)
CACHE4 = ExtendedPolicy("cache4", memory=4)
ADAPTIVE2 = ExtendedPolicy("adaptive2", memory=2, memory_mode="adaptive")
MODULO = ExtendedPolicy("modulo8", memory_mode="modulo")
PLAN = ExtendedPolicy("precomputed_plan", memory_mode="plan")
STALE = ExtendedPolicy("stale8", memory_mode="stale")
SCAN1 = ExtendedPolicy("scan1", scan=1)
SCAN2 = ExtendedPolicy("scan2", scan=2)
SCAN4 = ExtendedPolicy("scan4", scan=4)
FIXED4 = ExtendedPolicy("scan4_fixed", scan=4, selection="fixed")
PADDED = ExtendedPolicy("scan1_padded4", scan=1, selection="padded")
INDEXED = ExtendedPolicy("indexed", selection="indexed")
RANDOM = ExtendedPolicy("random_order", scheduler="random")
UNIFORM = ExtendedPolicy("uniform", uniform=True)
ELIMINATE = ExtendedPolicy("uniform_eliminate", uniform=True, eliminate=True)
CACHE_ELIMINATE = ExtendedPolicy("cache2_eliminate", memory=2, eliminate=True)
SCAN_ELIMINATE = ExtendedPolicy("scan1_eliminate", scan=1, eliminate=True)


def ready_order(task):
    done, order = set(), []
    while len(order) < task.config.n:
        available = [i for i, node in enumerate(task.nodes)
                     if i not in done and set(node.prerequisites) <= done]
        if not available:
            raise ValueError("Task graph is cyclic")
        node = min(available)
        order.append(node)
        done.add(node)
    return order


def prepare(config, seed, variant):
    task = generate(config, seed)
    order = ready_order(task)
    nodes = list(task.nodes)
    for position, node_id in enumerate(order):
        node = nodes[node_id]
        query = node.query
        if variant.queries == "clustered":
            query = position // (config.n // config.width)
        elif variant.queries == "concentrated":
            query = 0
        nodes[node_id] = replace(
            node, query=query,
            delta=0 if variant.updates == "static" else node.delta)
    if variant.reverse_labels:
        mapping = {i: config.n - 1 - i for i in range(config.n)}
        renamed = [None] * config.n
        for i, node in enumerate(nodes):
            renamed[mapping[i]] = replace(
                node, prerequisites=tuple(mapping[p] for p in node.prerequisites))
        nodes = renamed
    return Task(config, seed, task.initial, tuple(nodes))


class ExtendedEnvironment:
    def __init__(self, task, variant):
        self.task, self.variant = task, variant
        self._state = list(task.initial)
        self._done = set()
        self._terminal = False
        self._rng = random.Random(task.seed + 8000003)
        self.calls = self.errors = 0

    def observe(self):
        period = self.variant.snapshot_every
        visible = not self.calls or (period > 0 and len(self._done) % period == 0)
        return {
            "state": tuple(self._state) if visible else (None,) * 8,
            "done": tuple(sorted(self._done)),
            "terminal": self._terminal,
            "success": len(self._done) == self.task.config.n,
            "update": None, "reason": None, "resampled": False,
            "last_node": None, "last_code": None,
        }

    def execute(self, node_id, slot):
        if self._terminal:
            raise RuntimeError("Cannot execute after termination")
        if type(node_id) is not int or not 0 <= node_id < self.task.config.n:
            raise ValueError("Invalid operation identifier")
        node = self.task.nodes[node_id]
        if type(slot) is not int or not 0 <= slot < len(node.options):
            raise ValueError("Invalid candidate slot")
        self.calls += 1
        reason = None
        if node_id in self._done or not set(node.prerequisites) <= self._done:
            reason = "dependency"
        elif node.options[slot] != self._state[node.query] % self.task.config.branching:
            reason = "candidate"
        if reason:
            self.errors += 1
            self._terminal = self.errors > self.variant.retries
        else:
            self._done.add(node_id)
            self._state[node.update] = (self._state[node.update] + node.delta) % 8
            if self.variant.hidden_resampling:
                fresh = [self._rng.randrange(8) for _ in range(8)]
                self._state[:self.task.config.width] = fresh[:self.task.config.width]
            self._terminal = len(self._done) == self.task.config.n
        result = self.observe()
        result.update(reason=reason, last_node=node_id, last_code=node.options[slot])
        if not reason:
            result["update"] = (node.update, node.delta)
            result["resampled"] = self.variant.hidden_resampling
        return result


class ExtendedController:
    def __init__(self, task, variant, policy, seed):
        if policy.memory_mode == "plan" and variant.hidden_resampling:
            raise ValueError("A deterministic plan is not valid under hidden resampling")
        self.task, self.variant, self.policy = task, variant, policy
        self.rng = random.Random(seed)
        self.order = ready_order(task)
        self.cache = OrderedDict()
        self.forbidden = {}
        self.plan = {}
        self.initialized = False
        self.evaluations = self.lookups = self.preparation_units = 0
        self.max_cache_entries = self.max_cache_bits = 0
        self.snapshot_uses = 0
        self.indexes = ()
        if policy.selection == "indexed":
            self.indexes = tuple({code: slot for slot, code in enumerate(node.options)}
                                 for node in task.nodes)
            self.preparation_units = task.config.n * task.config.branching

    def _future_registers(self, done, skip_current=False):
        remaining = [i for i in self.order if i not in done]
        if skip_current:
            remaining = remaining[1:]
        return list(dict.fromkeys(self.task.nodes[i].query for i in remaining))

    def _trim_adaptive(self, done):
        future = self._future_registers(done)
        ranking = {register: index for index, register in enumerate(future)}
        ordered = sorted(self.cache, key=lambda r: ranking.get(r, self.task.config.n))
        self.cache = OrderedDict((r, self.cache[r]) for r in ordered[:self.policy.memory])

    def _load_snapshot(self, state, done):
        mode = self.policy.memory_mode
        registers = (self._future_registers(done, skip_current=True)
                     if mode == "adaptive" else list(range(self.task.config.width)))
        modulus = self.task.config.branching if mode in ("adaptive", "modulo") else 8
        self.cache = OrderedDict(
            (r, state[r] % modulus) for r in registers[:self.policy.memory])

    def _process(self, observation):
        done = set(observation["done"])
        mode = self.policy.memory_mode
        if observation["resampled"]:
            self.cache.clear()
        elif observation["last_node"] is not None and observation["reason"] is None:
            node = self.task.nodes[observation["last_node"]]
            if mode == "adaptive":
                self.cache[node.query] = observation["last_code"]
            if observation["update"] is not None and mode != "stale":
                register, delta = observation["update"]
                if register in self.cache:
                    modulus = (self.task.config.branching
                               if mode in ("adaptive", "modulo") else 8)
                    self.cache[register] = (self.cache[register] + delta) % modulus
            if mode == "adaptive":
                self._trim_adaptive(done)
        if self.policy.eliminate and observation["reason"] == "candidate":
            self.forbidden.setdefault(observation["last_node"], set()).add(
                observation["last_code"])
        visible = observation["state"][0] is not None
        if visible and mode != "plan":
            self._load_snapshot(observation["state"], done)
        if not self.initialized and mode == "plan":
            state = list(observation["state"])
            for node_id in self.order:
                node = self.task.nodes[node_id]
                self.plan[node_id] = node.options.index(
                    state[node.query] % self.task.config.branching)
                state[node.update] = (state[node.update] + node.delta) % 8
            self.preparation_units = self.task.config.n * self.task.config.branching
        self.initialized = True
        self.max_cache_entries = max(self.max_cache_entries, len(self.cache))
        bits = (self.task.config.branching.bit_length() - 1
                if mode in ("adaptive", "modulo") else 3)
        self.max_cache_bits = max(self.max_cache_bits, bits * len(self.cache))

    def act(self, observation):
        self._process(observation)
        done = set(observation["done"])
        candidates = [i for i, node in enumerate(self.task.nodes)
                      if i not in done and
                      (self.policy.scheduler == "random" or
                       set(node.prerequisites) <= done)]
        if not candidates:
            raise RuntimeError("No scheduling candidate")
        node_id = (self.rng.choice(candidates)
                   if self.policy.scheduler == "random" else min(candidates))
        node = self.task.nodes[node_id]
        if self.policy.memory_mode == "plan":
            self.lookups += 1
            return node_id, self.plan[node_id]
        blocked = self.forbidden.get(node_id, set())
        slots = [s for s, code in enumerate(node.options) if code not in blocked]
        if not slots:
            raise RuntimeError("All candidates rejected without a valid transition")
        value = observation["state"][node.query]
        if value is not None:
            self.snapshot_uses += 1
        else:
            value = self.cache.get(node.query)
        if value is None or self.policy.uniform:
            return node_id, self.rng.choice(slots)
        target = value % self.task.config.branching
        if self.policy.selection == "indexed":
            self.lookups += 1
            return node_id, self.indexes[node_id][target]
        limit = min(self.policy.scan, len(slots))
        examined = slots[:limit]
        if self.policy.selection == "padded":
            charged = slots[:min(4, len(slots))]
            for slot in charged:
                _ = node.options[slot] == target
            self.evaluations += len(charged)
        elif self.policy.selection == "fixed":
            self.evaluations += len(examined)
        match = None
        for slot in examined:
            if self.policy.selection == "serial":
                self.evaluations += 1
            if node.options[slot] == target:
                match = slot
                break
        if match is not None:
            return node_id, match
        fallback = slots[limit:] or slots
        return node_id, self.rng.choice(fallback)


def extended_rollout(task, variant, policy, seed, trace=False):
    env = ExtendedEnvironment(task, variant)
    controller = ExtendedController(task, variant, policy, seed)
    observation = env.observe()
    records = []
    reasons = {"dependency": 0, "candidate": 0}
    while not observation["terminal"]:
        action = controller.act(observation)
        following = env.execute(*action)
        if following["reason"]:
            reasons[following["reason"]] += 1
        if trace:
            records.append({"observation": observation, "action": action,
                            "result": following})
        observation = following
    return {
        "success": int(observation["success"]), "completed": len(observation["done"]),
        "calls": env.calls, "errors": env.errors,
        "evaluations": controller.evaluations, "lookups": controller.lookups,
        "preparation_units": controller.preparation_units,
        "max_cache_entries": controller.max_cache_entries,
        "max_cache_bits": controller.max_cache_bits,
        "plan_slots": len(controller.plan), "snapshot_uses": controller.snapshot_uses,
        "dependency_errors": reasons["dependency"],
        "candidate_errors": reasons["candidate"],
        **({"trace": records} if trace else {}),
    }


def retry_probability(n, branching, retries, eliminate):
    if not eliminate:
        p = 1 / branching
        return sum(comb(n+r-1, r) * p**n * (1-p)**r for r in range(retries+1))
    distribution = [1.0] + [0.0] * retries
    for _ in range(n):
        following = [0.0] * (retries+1)
        for failures, mass in enumerate(distribution):
            for extra in range(min(branching, retries-failures+1)):
                following[failures+extra] += mass / branching
        distribution = following
    return sum(distribution)


def predicted_success(task, variant, policy):
    c = task.config
    if policy.uniform:
        return retry_probability(c.n, c.branching, variant.retries, policy.eliminate)
    if variant.retries:
        return None
    if policy.scheduler == "random":
        from benchmark import expected_success, Policy
        return expected_success(c, Policy("random", scheduler="random"))
    if policy.memory_mode == "plan":
        return 1.0
    if variant.hidden_resampling:
        if policy.memory < c.width:
            return None
        visible = (c.n + variant.snapshot_every - 1) // variant.snapshot_every \
            if variant.snapshot_every else 1
        p = min(1.0, (policy.scan+1) / c.branching)
        return p**visible * c.branching**(-(c.n-visible))
    if policy.memory_mode not in ("fixed", "modulo"):
        return None
    state = list(task.initial)
    probability = 1.0
    for position, node_id in enumerate(ready_order(task)):
        node = task.nodes[node_id]
        visible = (position == 0 or
                   (variant.snapshot_every and position % variant.snapshot_every == 0)
                   or node.query < policy.memory)
        if not visible:
            probability /= c.branching
        elif policy.selection != "indexed":
            slot = node.options.index(state[node.query] % c.branching)
            limit = min(policy.scan, c.branching)
            if slot >= limit:
                probability /= c.branching-limit
        state[node.update] = (state[node.update]+node.delta) % 8
    return probability


@dataclass(frozen=True)
class Case:
    study: str
    level: str
    config: Config
    variant: Variant
    policy: ExtendedPolicy


def cases():
    for family in ("chains", "barriers"):
        for depth in (2, 4):
            for b in (4, 8):
                for period in (0, 1):
                    for n in (4, 8, 16):
                        c = Config(family, depth, b, 4, "history", n=n)
                        for policy in (EXACT, CACHE2, SCAN4, RANDOM):
                            yield Case("work", f"n{n}_p{period}", c,
                                       Variant(snapshot_every=period), policy)
        for b in (4, 8):
            for w in (4, 8):
                c = Config(family, 4, b, w, "history")
                for period in (0, 4):
                    for policy in (CACHE2, CACHE4, ADAPTIVE2, MODULO, PLAN, STALE):
                        yield Case("memory", f"p{period}", replace(c, n=16),
                                   Variant(snapshot_every=period), policy)
                for period in (0, 1, 2, 4):
                    for shock in (False, True):
                        for policy in (EXACT, CACHE2, SCAN4, UNIFORM):
                            yield Case("observation", f"p{period}_shock{int(shock)}",
                                       c, Variant(snapshot_every=period,
                                                  hidden_resampling=shock), policy)
                for period in (0, 1):
                    for retries in (0, 2, 8, 56):
                        for policy in (UNIFORM, ELIMINATE, CACHE_ELIMINATE,
                                       SCAN_ELIMINATE):
                            yield Case("recovery", f"r{retries}_p{period}", c,
                                       Variant(snapshot_every=period, retries=retries),
                                       policy)
                    for reverse in (False, True):
                        for policy in (EXACT, CACHE2, SCAN4, RANDOM):
                            yield Case("labels", f"reverse{int(reverse)}_p{period}",
                                       c, Variant(snapshot_every=period,
                                                  reverse_labels=reverse), policy)
                for queries in ("balanced", "clustered", "concentrated"):
                    for updates in ("additive", "static"):
                        for policy in (CACHE2, ADAPTIVE2, EXACT, STALE):
                            yield Case("locality", f"{queries}_{updates}",
                                       replace(c, n=16),
                                       Variant(queries=queries, updates=updates),
                                       policy)
        for b in (2, 4, 8):
            c = Config(family, 4, b, 4, "history")
            for policy in (SCAN1, SCAN2, FIXED4, PADDED, EXACT, INDEXED):
                yield Case("selection", f"b{b}", c, Variant(snapshot_every=1), policy)
