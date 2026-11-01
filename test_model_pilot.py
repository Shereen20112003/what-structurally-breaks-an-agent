import unittest

from run_model_pilot import intervention_configs, parse_action


class ModelPilotTests(unittest.TestCase):
    def test_intervention_grid_is_paired_and_one_factor_at_a_time(self):
        configs = list(intervention_configs())
        self.assertEqual(len(configs), 8)
        by_factor = {}
        for factor, level, config in configs:
            by_factor.setdefault(factor, {})[level] = config
        self.assertEqual(set(by_factor), {
            "depth", "branching", "width", "observation"
        })
        for levels in by_factor.values():
            self.assertEqual(set(levels), {"low", "high"})

    def test_action_parser_requires_exact_integer_schema(self):
        self.assertEqual(parse_action('{"node_id":2,"slot":1}'), ((2, 1), None))
        self.assertEqual(parse_action('{"node_id":true,"slot":1}')[1],
                         "invalid_action_schema")
        self.assertEqual(parse_action('not json')[1], "invalid_json")
        self.assertEqual(parse_action('{"node_id":2,"slot":1,"why":"x"}')[1],
                         "invalid_action_schema")


if __name__ == "__main__":
    unittest.main()
