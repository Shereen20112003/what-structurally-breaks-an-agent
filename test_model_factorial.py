import unittest

from benchmark import Environment, generate
from run_model_factorial import factorial_conditions, replay_attempt


class ModelFactorialTests(unittest.TestCase):
    def test_fractional_factorial_has_orthogonal_balanced_main_effects(self):
        conditions = factorial_conditions()
        self.assertEqual(len(conditions), 8)
        factors = ("depth", "branching", "width", "observation")
        for factor in factors:
            self.assertEqual(
                sum(row["codes"][factor] for row in conditions), 0
            )
        for first_index, first in enumerate(factors):
            for second in factors[first_index + 1:]:
                self.assertEqual(
                    sum(row["codes"][first] * row["codes"][second]
                        for row in conditions),
                    0,
                )
        for row in conditions:
            codes = row["codes"]
            self.assertEqual(
                codes["observation"],
                codes["depth"] * codes["branching"] * codes["width"],
            )
            config = row["config"]
            self.assertEqual(config.n, 8)
            self.assertEqual(config.family, "chains")

    def test_replay_marks_invalid_output_terminal_without_an_extra_action(self):
        task = generate(factorial_conditions()[0]["config"], seed=20263)
        attempt = {"turns": [{
            "parse_error": "invalid_json",
            "action": None,
        }]}
        env, observation, history, stop_reason = replay_attempt(task, attempt)
        self.assertEqual(stop_reason, None)
        self.assertEqual(len(history), 1)
        self.assertTrue(observation["terminal"])
        self.assertFalse(observation["success"])
        self.assertEqual(observation["reason"], "invalid_json")
        self.assertEqual(env.calls, 0)

    def test_replay_reconstructs_successful_state_from_logged_actions(self):
        config = factorial_conditions()[0]["config"]
        task = generate(config, seed=20263)
        env = Environment(task)
        observation = env.observe()
        turns = []
        # A logged action sequence is replayed through the task environment.
        # The exact action values are found by an independent test controller.
        from benchmark import Controller, Policy
        controller = Controller(task, Policy("exact"), seed=0)
        while not observation["terminal"]:
            action = controller.act(observation)
            observation = env.execute(*action)
            turns.append({"action": list(action)})
        replayed_env, replayed_observation, _, stop_reason = replay_attempt(
            task, {"turns": turns}
        )
        self.assertIsNone(stop_reason)
        self.assertTrue(replayed_observation["success"])
        self.assertEqual(replayed_env.calls, config.n)


if __name__ == "__main__":
    unittest.main()
