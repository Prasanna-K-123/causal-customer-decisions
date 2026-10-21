"""Adversarial input and row-order invariance checks for causal decisions."""
import unittest
import numpy as np

from customer_policy.evaluation import (
    allocate, augmented_scores, paired_bootstrap, policy_scores, summarize,
    uniform_policy,
)


class InputContractTests(unittest.TestCase):
    def test_fractional_treatment_labels_are_not_truncated(self):
        for arm in ([.9, 1.9, 2.9], [0, .5, 2], [0, 1, np.nan],
                    [0, 1, np.inf], [0, 1, 3], [False, True, False],
                    ["0", "1", "2"]):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                augmented_scores([9, 12, 15], arm, np.zeros((3, 3)))

    def test_fractional_policy_actions_are_not_truncated(self):
        for actions in ([.9, 1.9, 2.9], [0, -1, 2], [0, 1, np.inf],
                        [0, 1, 2 + 1j], [True, False, True]):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                policy_scores(np.eye(3), actions)

    def test_nonfinite_outcomes_and_score_matrices_are_rejected(self):
        for bad in (np.nan, np.inf, -np.inf, 1 + 2j):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    augmented_scores([bad, 1], [0, 1], np.zeros((2, 3)))
                with self.assertRaises(ValueError):
                    policy_scores([[0, bad, 0]], [0])

    def test_dimension_and_alignment_errors_are_rejected(self):
        cases = [
            lambda: augmented_scores([[1], [2]], [0, 1], np.zeros((2, 3))),
            lambda: augmented_scores([1, 2], [[0], [1]], np.zeros((2, 3))),
            lambda: policy_scores(np.zeros((2, 2)), [0, 1]),
            lambda: policy_scores(np.zeros((2, 3)), [[0], [1]]),
            lambda: summarize(np.zeros((2, 2))),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                case()

    def test_nonfinite_scenario_parameters_are_rejected(self):
        for bad in (np.nan, np.inf, -np.inf, 1j, True, [.4]):
            for parameter in ("margin", "cost", "capacity"):
                arguments = {"margin": .4, "cost": .05, "capacity": .5}
                arguments[parameter] = bad
                with self.subTest(bad=bad, parameter=parameter), self.assertRaises(ValueError):
                    allocate(np.ones((3, 3)), tie_key=np.arange(3), **arguments)
            for parameter in ("margin", "cost"):
                arguments = {"margin": .4, "cost": .05}
                arguments[parameter] = bad
                with self.subTest(bad=bad, parameter=parameter), self.assertRaises(ValueError):
                    policy_scores(np.ones((3, 3)), [0, 1, 2], **arguments)

    def test_invalid_confidence_is_rejected(self):
        for confidence in (0, 1, -1, 1.1, np.nan, np.inf, [0.95], np.nextafter(1., 0.)):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                summarize([1, 2, 3], confidence)

    def test_invalid_bootstrap_samples_and_repetitions_are_rejected(self):
        for sample in ([], [1], [1, np.nan], [[1, 2], [3, 4]]):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                paired_bootstrap(sample, seed=7, repetitions=2)
        for repetitions in (0, 1, -1, 2.5, True):
            with self.subTest(repetitions=repetitions), self.assertRaises(ValueError):
                paired_bootstrap([1, 2, 3], seed=7, repetitions=repetitions)

    def test_uniform_policy_validates_arm_batch_capacity_priority(self):
        cases = [
            (.9, 3, .5, [0, 1, 2]), (3, 3, .5, [0, 1, 2]),
            (1, -1, .5, []), (1, 3.0, .5, [0, 1, 2]),
            (1, 3, np.nan, [0, 1, 2]), (1, 3, 1.5, [0, 1, 2]),
            (1, 3, .5, [0, 1]), (1, 3, .5, [0, 1, np.nan]),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                uniform_policy(*args)

    def test_tie_key_contract_prevents_ambiguous_priorities(self):
        for keys in ([0, 0, 1], [0, 1, np.nan], [[0], [1], [2]], [0, 1]):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                allocate(np.ones((3, 3)), .4, .05, .5, keys)

    def test_arithmetic_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            allocate([[-1e308, 1e308, 0]], 1, 0, 1, [0])
        with self.assertRaises(ValueError):
            augmented_scores([1e308], [0], [[0, 0, 0]], [1e-300, .5, .5])
        with self.assertRaises(ValueError):
            policy_scores([[0, -1e308, 0]], [1], cost=1e308, margin=1)
        with self.assertRaises(ValueError):
            summarize([-1e308, 1e308])

    def test_equal_value_ties_have_a_reproducible_capacity_cutoff(self):
        means = np.tile([0., 2., 1.], (5, 1))
        keys = np.array([50, 10, 40, 20, 30])
        actions = allocate(means, 1, 0, .5, keys)
        np.testing.assert_array_equal(actions, [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(uniform_policy(2, 5, .5, keys), [0, 2, 0, 2, 0])

    def test_joint_row_permutation_preserves_scores_and_assignments(self):
        rng = np.random.default_rng(20261021)
        y = rng.normal(size=43)
        arm = rng.integers(0, 3, len(y))
        mu = rng.normal(size=(len(y), 3))
        keys = rng.permutation(len(y))
        permutation = rng.permutation(len(y))
        gamma = augmented_scores(y, arm, mu)
        permuted = augmented_scores(y[permutation], arm[permutation], mu[permutation])
        np.testing.assert_array_equal(permuted, gamma[permutation])
        actions = allocate(mu, .4, .05, .4, keys)
        other = allocate(mu[permutation], .4, .05, .4, keys[permutation])
        np.testing.assert_array_equal(other, actions[permutation])
        scores = policy_scores(gamma, actions, .4, .05)
        permuted_scores = policy_scores(permuted, other, .4, .05)
        np.testing.assert_array_equal(permuted_scores, scores[permutation])
        self.assertAlmostEqual(summarize(scores)["estimate"], summarize(permuted_scores)["estimate"])


if __name__ == "__main__":
    unittest.main()
