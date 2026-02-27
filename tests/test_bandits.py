"""
test_bandits.py
────────────────
Unit tests for all three bandit algorithm implementations.

Run with:
    pytest tests/test_bandits.py -v
"""

import pytest
from app.bandits.epsilon_greedy import EpsilonGreedy
from app.bandits.ucb import UCB
from app.bandits.thompson_sampling import ThompsonSampling
from app.bandits.factory import create_bandit


# ─── EpsilonGreedy ───────────────────────────────────────────────────────────

class TestEpsilonGreedy:
    def test_select_arm_returns_valid_index(self):
        bandit = EpsilonGreedy(n_arms=3, epsilon=0.1)
        for _ in range(50):
            arm = bandit.select_arm()
            assert 0 <= arm < 3

    def test_update_increments_count(self):
        bandit = EpsilonGreedy(n_arms=3)
        bandit.update(0, 1.0)
        assert bandit.counts[0] == 1

    def test_update_computes_running_average(self):
        bandit = EpsilonGreedy(n_arms=3)
        bandit.update(0, 1.0)
        bandit.update(0, 0.0)
        assert bandit.values[0] == pytest.approx(0.5)

    def test_exploits_best_arm_with_epsilon_zero(self):
        bandit = EpsilonGreedy(n_arms=3, epsilon=0.0)
        # Arm 2 is best
        bandit.update(2, 1.0)
        for _ in range(10):
            assert bandit.select_arm() == 2

    def test_state_roundtrip(self):
        bandit = EpsilonGreedy(n_arms=3)
        bandit.update(1, 0.8)
        state = bandit.get_state()
        new_bandit = EpsilonGreedy(n_arms=3)
        new_bandit.load_state(state)
        assert new_bandit.values == bandit.values
        assert new_bandit.counts == bandit.counts


# ─── UCB ─────────────────────────────────────────────────────────────────────

class TestUCB:
    def test_select_arm_returns_valid_index(self):
        bandit = UCB(n_arms=3)
        for _ in range(50):
            # Must update to avoid always pulling unpulled arms
            arm = bandit.select_arm()
            bandit.update(arm, 1.0)
            assert 0 <= arm < 3

    def test_pulls_each_arm_at_least_once_first(self):
        bandit = UCB(n_arms=3)
        pulled = set()
        for _ in range(3):
            arm = bandit.select_arm()
            bandit.update(arm, 1.0)
            pulled.add(arm)
        assert pulled == {0, 1, 2}

    def test_update_increments_total_count(self):
        bandit = UCB(n_arms=3)
        bandit.update(0, 1.0)
        bandit.update(1, 0.5)
        assert bandit.total_counts == 2

    def test_state_roundtrip(self):
        bandit = UCB(n_arms=3)
        bandit.update(0, 1.0)
        bandit.update(1, 0.5)
        state = bandit.get_state()
        new_bandit = UCB(n_arms=3)
        new_bandit.load_state(state)
        assert new_bandit.counts == bandit.counts
        assert new_bandit.total_counts == bandit.total_counts


# ─── ThompsonSampling ────────────────────────────────────────────────────────

class TestThompsonSampling:
    def test_select_arm_returns_valid_index(self):
        bandit = ThompsonSampling(n_arms=3)
        for _ in range(50):
            arm = bandit.select_arm()
            assert 0 <= arm < 3

    def test_update_increments_alpha_on_success(self):
        bandit = ThompsonSampling(n_arms=3)
        bandit.update(1, 1)
        assert bandit.alpha[1] == 2  # starts at 1, +1 for success

    def test_update_increments_beta_on_failure(self):
        bandit = ThompsonSampling(n_arms=3)
        bandit.update(1, 0)
        assert bandit.beta[1] == 2  # starts at 1, +1 for failure

    def test_converges_to_best_arm(self):
        """After many updates, the best arm should be selected most often."""
        bandit = ThompsonSampling(n_arms=3)
        # Simulate: arm 1 always wins, others always lose
        for _ in range(200):
            bandit.update(0, 0)
            bandit.update(1, 1)
            bandit.update(2, 0)

        counts = [0, 0, 0]
        for _ in range(1000):
            arm = bandit.select_arm()
            counts[arm] += 1

        # Arm 1 should dominate
        assert counts[1] > counts[0]
        assert counts[1] > counts[2]

    def test_state_roundtrip(self):
        bandit = ThompsonSampling(n_arms=3)
        bandit.update(0, 1)
        bandit.update(0, 1)
        state = bandit.get_state()
        new_bandit = ThompsonSampling(n_arms=3)
        new_bandit.load_state(state)
        assert new_bandit.alpha == bandit.alpha
        assert new_bandit.beta == bandit.beta


# ─── Factory ─────────────────────────────────────────────────────────────────

class TestFactory:
    def test_creates_thompson(self):
        b = create_bandit("thompson", 3)
        assert isinstance(b, ThompsonSampling)

    def test_creates_epsilon_greedy(self):
        b = create_bandit("epsilon_greedy", 3, epsilon=0.2)
        assert isinstance(b, EpsilonGreedy)
        assert b.epsilon == 0.2

    def test_creates_ucb(self):
        b = create_bandit("ucb", 3)
        assert isinstance(b, UCB)

    def test_raises_on_unknown_algorithm(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_bandit("fake_algo", 3)