import unittest
from dataclasses import replace
from itertools import permutations
from math import factorial

from benchmark import (Config, Controller, Environment, POLICIES, generate,
                       grid, rollout, structural_metrics, task_expected_success)


class WorkflowTests(unittest.TestCase):
    def test_exact_controller_solves_entire_grid_from_observations(self):
        for config in grid():
            for seed in range(8):
                result = rollout(generate(config, seed), POLICIES[0], seed)
                self.assertEqual(result["success"], 1, (config, seed))
                self.assertEqual(result["calls"], config.n)

    def test_target_metrics_and_preserved_optimal_work(self):
        for config in grid():
            metrics = structural_metrics(generate(config, 31))
            self.assertEqual(metrics["critical_path"], config.depth)
            self.assertEqual(metrics["active_registers"], config.width)
            self.assertEqual(metrics["candidate_count"], config.branching)
            self.assertEqual(metrics["optimal_calls"], 8)
            k = 8 // config.depth
            expected = (factorial(8) // factorial(config.depth) ** k
                        if config.family == "chains"
                        else factorial(k) ** config.depth)
            self.assertEqual(metrics["linear_extensions"], expected)

    def test_exhaustive_small_graph_matches_count(self):
        for family in ("chains", "barriers"):
            task = generate(Config(family, 2, 2, 2, "history", n=4), 5)
            count = 0
            for order in permutations(range(4)):
                done = set()
                valid = True
                for node in order:
                    if not set(task.nodes[node].prerequisites) <= done:
                        valid = False
                        break
                    done.add(node)
                count += valid
            self.assertEqual(count, structural_metrics(task)["linear_extensions"])

    def test_hidden_feedback_still_has_public_delta(self):
        task = generate(Config("chains", 8, 4, 4, "history"), 7)
        env = Environment(task)
        controller = Controller(task, POLICIES[0], 8)
        result = env.execute(*controller.act(env.observe()))
        self.assertEqual(result["state"], (None,) * 8)
        self.assertIsNotNone(result["update"])
        self.assertEqual(result["reason"], None)

    def test_wrong_action_is_terminal_and_explicit(self):
        task = generate(Config("chains", 8, 2, 1, "snapshot"), 1)
        env = Environment(task)
        controller = Controller(task, POLICIES[0], 0)
        node, option = controller.act(env.observe())
        result = env.execute(node, 1 - option)
        self.assertEqual(result["reason"], "candidate")
        self.assertTrue(result["terminal"])
        with self.assertRaises(RuntimeError):
            env.execute(node, option)

    def test_dependency_violation_and_invalid_inputs(self):
        task = generate(Config("chains", 8, 2, 1, "snapshot"), 1)
        env = Environment(task)
        dependent = next(i for i, node in enumerate(task.nodes)
                         if node.prerequisites)
        with self.assertRaises(ValueError):
            env.execute(-1, 0)
        self.assertEqual(env.calls, 0)
        self.assertEqual(env.execute(dependent, 0)["reason"], "dependency")

    def test_graph_channel_pairing_keeps_content_identical(self):
        config = Config("chains", 4, 4, 4, "snapshot")
        first = generate(config, 19)
        for other in (replace(config, family="barriers"),
                      replace(config, observation="history")):
            second = generate(other, 19)
            self.assertEqual(first.initial, second.initial)
            for a, b in zip(first.nodes, second.nodes):
                self.assertEqual((a.query, a.update, a.delta, a.options),
                                 (b.query, b.update, b.delta, b.options))

    def test_snapshot_removes_fixed_cache_deficit(self):
        for seed in range(32):
            task = generate(Config("barriers", 4, 8, 8, "snapshot"), seed)
            self.assertEqual(rollout(task, POLICIES[1], seed)["success"], 1)

    def test_unknown_query_bound_accounts_for_visible_first_action(self):
        task = generate(Config("chains", 8, 2, 8, "history"), 25)
        node = next(node for node in task.nodes if not node.prerequisites)
        unknown = 6 - int(node.query >= 2)
        self.assertEqual(task_expected_success(task, POLICIES[1]), 2 ** -unknown)

    def test_scan_limit_changes_choices_and_evaluation_cost(self):
        task = generate(Config("chains", 1, 8, 1, "snapshot"), 9)
        result = rollout(task, POLICIES[3], 10, trace=True)
        self.assertLessEqual(result["evaluations"], result["calls"])
        self.assertEqual(len(result["trace"]), result["calls"])

    def test_conditional_scan_prediction_uses_actual_candidate_positions(self):
        task = generate(Config("chains", 1, 8, 1, "snapshot", n=1), 9)
        node = task.nodes[0]
        for slot in range(8):
            target = node.options[slot]
            changed = replace(task, initial=(target,) + task.initial[1:])
            self.assertEqual(task_expected_success(changed, POLICIES[3]),
                             1 if slot == 0 else 1 / 7)


if __name__ == "__main__":
    unittest.main()
