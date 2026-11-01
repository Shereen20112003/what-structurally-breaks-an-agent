import unittest

from analyze_local_study import analyze_episodes
from run_local_study import (
    EpisodeExecutionError,
    FACTORS,
    SEEDS,
    factorial_cases,
    parse_action,
    run_episode,
)


class LocalStudyTests(unittest.TestCase):
    def test_frozen_grid_is_complete_factorial_with_fixed_work(self):
        cases = list(factorial_cases())
        self.assertEqual(len(cases), 2 ** len(FACTORS))
        self.assertEqual(len(cases) * len(SEEDS), 256)
        self.assertEqual({config.n for _, config in cases}, {8})
        self.assertEqual(
            {(family, config.depth, config.branching, config.width,
              config.observation) for family, config in cases},
            {
                (family, depth, branching, width, observation)
                for family in FACTORS["family"]
                for depth in FACTORS["depth"]
                for branching in FACTORS["branching"]
                for width in FACTORS["width"]
                for observation in FACTORS["observation"]
            },
        )

    def test_action_schema_is_strict(self):
        self.assertEqual(parse_action('{"node_id":1,"slot":0}'), ((1, 0), None))
        self.assertEqual(parse_action('{"node_id":true,"slot":0}')[1],
                         "invalid_action_schema")
        self.assertEqual(parse_action('{"node_id":1,"slot":0,"extra":2}')[1],
                         "invalid_action_schema")
        self.assertEqual(parse_action("not json")[1], "invalid_json")

    def test_analysis_recovers_a_known_paired_factor_effect(self):
        episodes = []
        for family, config in factorial_cases():
            for seed in SEEDS:
                episodes.append({
                    "seed": seed,
                    "config": {
                        "family": family,
                        "depth": config.depth,
                        "branching": config.branching,
                        "width": config.width,
                        "observation": config.observation,
                        "n": 8,
                    },
                    "factorial_cell": {
                        "family": family,
                        "depth": config.depth,
                        "branching": config.branching,
                        "width": config.width,
                        "observation": config.observation,
                    },
                    "success": int(config.branching == 8),
                    "completed": 8 if config.branching == 8 else 0,
                    "decisions": [],
                    "first_failure": "none",
                })
        summary = analyze_episodes(episodes)
        branching = next(
            item for item in summary["contrasts"]
            if item["factor"] == "branching"
        )
        self.assertEqual(summary["episode_count"], 256)
        self.assertEqual(branching["paired_difference"], 1)
        self.assertEqual(
            branching["seed_cluster_bootstrap_95_percent_interval"],
            [1, 1],
        )

    def test_analysis_rejects_an_incomplete_grid(self):
        with self.assertRaisesRegex(ValueError, "256 unique"):
            analyze_episodes([])

    def test_interrupted_inference_is_not_a_completed_episode(self):
        class FailingModel:
            def create_chat_completion(self, **kwargs):
                raise RuntimeError("simulated local runtime failure")

        family, config = next(factorial_cases())
        with self.assertRaises(EpisodeExecutionError) as caught:
            run_episode(FailingModel(), family, config, SEEDS[0])
        partial = caught.exception.partial_episode
        self.assertFalse(partial["attempt_complete"])
        self.assertEqual(partial["first_failure"], "inference_error")
        self.assertEqual(partial["decisions"], [])


if __name__ == "__main__":
    unittest.main()
