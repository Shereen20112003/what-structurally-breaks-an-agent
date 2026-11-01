import unittest
from dataclasses import replace
from itertools import product

from benchmark import POLICIES, Config, generate, rollout, structural_metrics
from extended_benchmark import (
    ADAPTIVE2, CACHE2, EXACT, FIXED4, INDEXED, MODULO, PADDED, PLAN, STALE,
    UNIFORM, ELIMINATE, ExtendedController, ExtendedEnvironment, Variant,
    cases, extended_rollout, predicted_success, prepare, ready_order,
    retry_probability)


class ExtendedTests(unittest.TestCase):
    def test_original_reference_behavior_is_preserved(self):
        for period in (0, 1):
            c = Config("chains", 4, 8, 8,
                       "snapshot" if period else "history")
            v = Variant(snapshot_every=period)
            for seed in range(16):
                task = generate(c, seed)
                for original, policy in ((POLICIES[0], EXACT), (POLICIES[1], CACHE2)):
                    old, new = rollout(task, original, seed), extended_rollout(
                        task, v, policy, seed)
                    for key in ("success", "completed", "calls", "evaluations"):
                        self.assertEqual(old[key], new[key], (seed, key))

    def test_exact_modulo_and_preplanning_solve_deterministic_variants(self):
        for n, period, queries, updates in product(
                (4, 8, 16), (0, 1, 4), ("balanced", "clustered"), ("additive", "static")):
            c = Config("barriers", 4, 8, 4, "history", n=n)
            v = Variant(snapshot_every=period, queries=queries, updates=updates)
            task = prepare(c, 21, v)
            for policy in (EXACT, MODULO, PLAN, INDEXED):
                result = extended_rollout(task, v, policy, 51)
                self.assertEqual(result["success"], 1, (n, period, queries, policy))

    def test_retry_is_state_preserving_and_returns_failure(self):
        c = Config("chains", 4, 4, 4, "history")
        v = Variant(retries=2)
        task = prepare(c, 5, v)
        env = ExtendedEnvironment(task, v)
        controller = ExtendedController(task, v, EXACT, 4)
        node, slot = controller.act(env.observe())
        failed = env.execute(node, (slot+1) % 4)
        self.assertFalse(failed["terminal"])
        self.assertEqual(failed["reason"], "candidate")
        self.assertEqual(failed["done"], ())
        self.assertIsNone(failed["update"])
        self.assertEqual(env.execute(node, slot)["reason"], None)

    def test_error_budget_exhaustion_and_invalid_arguments(self):
        task = prepare(Config("chains", 4, 2, 1, "history"), 6, Variant(retries=1))
        env = ExtendedEnvironment(task, Variant(retries=1))
        with self.assertRaises(ValueError):
            env.execute(True, 0)
        node = ready_order(task)[0]
        wrong = 1-task.nodes[node].options.index(task.initial[0] % 2)
        self.assertFalse(env.execute(node, wrong)["terminal"])
        self.assertTrue(env.execute(node, wrong)["terminal"])
        with self.assertRaises(RuntimeError):
            env.execute(node, wrong)

    def test_elimination_guarantees_success_with_worst_case_retry_allowance(self):
        c = Config("barriers", 4, 8, 8, "history")
        v = Variant(retries=56)
        for seed in range(32):
            task = prepare(c, seed, v)
            result = extended_rollout(task, v, ELIMINATE, seed)
            self.assertEqual(result["success"], 1)
            self.assertLessEqual(result["calls"], 64)

    def test_negative_binomial_and_elimination_calibrations(self):
        self.assertAlmostEqual(retry_probability(1, 4, 2, False), 1-(3/4)**3)
        self.assertAlmostEqual(retry_probability(1, 4, 2, True), 3/4)
        self.assertAlmostEqual(retry_probability(4, 4, 12, True), 1)
        self.assertAlmostEqual(retry_probability(8, 8, 0, False), 8**-8)

    def test_hidden_resampling_invalidates_history_cache(self):
        c = Config("chains", 4, 4, 4, "history")
        v = Variant(hidden_resampling=True)
        task = prepare(c, 21, v)
        env = ExtendedEnvironment(task, v)
        controller = ExtendedController(task, v, EXACT, 3)
        following = env.execute(*controller.act(env.observe()))
        self.assertTrue(following["resampled"])
        self.assertEqual(following["state"], (None,)*8)
        controller.act(following)
        self.assertFalse(controller.cache)
        with self.assertRaises(ValueError):
            ExtendedController(task, v, PLAN, 3)

    def test_snapshot_cadence_and_information_bound(self):
        c = Config("chains", 4, 4, 4, "history")
        for period, exponent in ((0, 7), (1, 0), (2, 4), (4, 6)):
            v = Variant(snapshot_every=period, hidden_resampling=True)
            task = prepare(c, 10, v)
            self.assertEqual(predicted_success(task, v, EXACT), 4**-exponent)
            if period == 1:
                self.assertEqual(extended_rollout(task, v, EXACT, 5)["success"], 1)

    def test_clustered_queries_follow_execution_order_and_reduce_active_width_only_when_concentrated(self):
        c = Config("chains", 4, 4, 4, "history", n=16)
        task = prepare(c, 11, Variant(queries="clustered"))
        queries = [task.nodes[i].query for i in ready_order(task)]
        self.assertEqual(queries, [0]*4+[1]*4+[2]*4+[3]*4)
        concentrated = prepare(c, 11, Variant(queries="concentrated"))
        self.assertEqual(len({n.query for n in concentrated.nodes}), 1)

    def test_sham_comparisons_change_cost_not_decisions(self):
        c = Config("barriers", 4, 8, 4, "history")
        v = Variant(snapshot_every=1)
        from extended_benchmark import SCAN1
        for seed in range(32):
            task = prepare(c, seed, v)
            baseline = extended_rollout(task, v, SCAN1, seed, trace=True)
            padded = extended_rollout(task, v, PADDED, seed, trace=True)
            self.assertEqual(baseline["trace"], padded["trace"])
            self.assertEqual(padded["evaluations"], 4*padded["calls"])
            fixed = extended_rollout(task, v, FIXED4, seed)
            self.assertEqual(fixed["evaluations"], 4*fixed["calls"])

    def test_index_preparation_and_modulo_cache_storage_are_accounted(self):
        c = Config("chains", 4, 4, 4, "history")
        v = Variant()
        task = prepare(c, 3, v)
        indexed = extended_rollout(task, v, INDEXED, 2)
        modulo = extended_rollout(task, v, MODULO, 2)
        self.assertEqual(indexed["preparation_units"], 32)
        self.assertEqual(indexed["lookups"], 8)
        self.assertEqual(modulo["max_cache_bits"], 8)

    def test_adaptive_cache_capacity_and_static_negative_control(self):
        c = Config("chains", 4, 4, 4, "history", n=16)
        for queries in ("balanced", "clustered"):
            v = Variant(queries=queries, updates="static")
            task = prepare(c, 12, v)
            self.assertEqual(extended_rollout(task, v, STALE, 2)["success"], 1)
            self.assertLessEqual(
                extended_rollout(task, v, ADAPTIVE2, 2)["max_cache_entries"], 2)

    def test_label_reversal_preserves_graph_audit_and_solvability(self):
        c = Config("barriers", 4, 4, 4, "history")
        original = prepare(c, 8, Variant())
        reverse = prepare(c, 8, Variant(reverse_labels=True))
        for metric in ("critical_path", "linear_extensions", "reachable_completion_masks"):
            self.assertEqual(structural_metrics(original)[metric],
                             structural_metrics(reverse)[metric])
        self.assertEqual(extended_rollout(reverse, Variant(), EXACT, 2)["success"], 1)

    def test_case_grid_and_variant_errors(self):
        design = list(cases())
        self.assertEqual(len({repr(case) for case in design}), len(design))
        self.assertEqual({c.study for c in design},
                         {"work", "memory", "observation", "recovery",
                          "labels", "locality", "selection"})
        with self.assertRaises(ValueError):
            Variant(snapshot_every=-1)
        with self.assertRaises(ValueError):
            Variant(queries="unknown")


if __name__ == "__main__":
    unittest.main()
