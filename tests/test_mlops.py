"""Tests for MLOps modules (tracking, registry, A/B testing)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fittrack.mlops.ab_testing import (
    ABTest,
    ABTestResult,
    TestStatus,
    TrafficSplitter,
    Variant,
    compute_sample_size,
    run_offline_ab_test,
)


class TestTrafficSplitter:
    """Tests for TrafficSplitter."""

    def test_equal_split(self) -> None:
        """Test 50/50 traffic split."""
        splitter = TrafficSplitter({"A": 0.5, "B": 0.5})

        assignments = [splitter.assign(f"user_{i}") for i in range(1000)]
        counts = {"A": assignments.count("A"), "B": assignments.count("B")}

        # Should be roughly 50/50
        assert 400 < counts["A"] < 600
        assert 400 < counts["B"] < 600

    def test_unequal_split(self) -> None:
        """Test 80/20 traffic split."""
        splitter = TrafficSplitter({"control": 0.8, "treatment": 0.2})

        assignments = [splitter.assign(f"user_{i}") for i in range(1000)]
        control_count = assignments.count("control")

        # Should be roughly 80/20
        assert 700 < control_count < 900

    def test_deterministic(self) -> None:
        """Test that same ID always gets same assignment."""
        splitter = TrafficSplitter({"A": 0.5, "B": 0.5})

        first = splitter.assign("user_123")
        second = splitter.assign("user_123")

        assert first == second

    def test_invalid_weights(self) -> None:
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            TrafficSplitter({"A": 0.3, "B": 0.3})

    def test_assign_batch(self) -> None:
        """Test batch assignment."""
        splitter = TrafficSplitter({"A": 0.5, "B": 0.5})

        ids = ["user_1", "user_2", "user_3"]
        assignments = splitter.assign_batch(ids)

        assert len(assignments) == 3
        assert all(a in ["A", "B"] for a in assignments)


class TestVariant:
    """Tests for Variant dataclass."""

    def test_variant_creation(self) -> None:
        """Test creating a variant."""
        model = LogisticRegression()
        variant = Variant(name="control", model=model, traffic_weight=0.5)

        assert variant.name == "control"
        assert variant.traffic_weight == 0.5
        assert variant.n_samples == 0

    def test_variant_sample_count(self) -> None:
        """Test prediction counting."""
        variant = Variant(name="test", model=None)
        variant.predictions = [0, 1, 1, 0, 1]

        assert variant.n_samples == 5


class TestABTest:
    """Tests for ABTest class."""

    @pytest.fixture
    def simple_models(self) -> tuple:
        """Create simple models for testing."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        model_a = LogisticRegression(max_iter=200)
        model_b = LogisticRegression(max_iter=200, C=0.5)

        model_a.fit(X, y)
        model_b.fit(X, y)

        return model_a, model_b

    def test_create_test(self) -> None:
        """Test creating an A/B test."""
        test = ABTest("my-test")

        assert test.name == "my-test"
        assert test.status == TestStatus.DRAFT
        assert len(test.variants) == 0

    def test_add_variants(self, simple_models: tuple) -> None:
        """Test adding variants."""
        model_a, model_b = simple_models
        test = ABTest("test")

        test.add_variant("control", model_a, 0.5)
        test.add_variant("treatment", model_b, 0.5)

        assert len(test.variants) == 2
        assert "control" in test.variants
        assert "treatment" in test.variants

    def test_start_stop(self, simple_models: tuple) -> None:
        """Test test lifecycle."""
        model_a, model_b = simple_models
        test = ABTest("test")
        test.add_variant("control", model_a, 0.5)
        test.add_variant("treatment", model_b, 0.5)

        test.start()
        assert test.status == TestStatus.RUNNING

        test.stop()
        assert test.status == TestStatus.STOPPED

    def test_cannot_add_after_start(self, simple_models: tuple) -> None:
        """Test that variants can't be added after start."""
        model_a, model_b = simple_models
        test = ABTest("test")
        test.add_variant("control", model_a, 0.5)
        test.add_variant("treatment", model_b, 0.5)
        test.start()

        with pytest.raises(RuntimeError):
            test.add_variant("new", model_a, 0.0)

    def test_predict(self, simple_models: tuple) -> None:
        """Test making predictions."""
        model_a, model_b = simple_models
        test = ABTest("test")
        test.add_variant("control", model_a, 0.5)
        test.add_variant("treatment", model_b, 0.5)
        test.start()

        X_sample = np.random.randn(1, 10)
        variant, pred = test.predict(X_sample, sample_id="user_1")

        assert variant in ["control", "treatment"]
        assert len(pred) == 1

    def test_record_outcome(self, simple_models: tuple) -> None:
        """Test recording outcomes."""
        model_a, model_b = simple_models
        test = ABTest("test")
        test.add_variant("control", model_a, 0.5)
        test.add_variant("treatment", model_b, 0.5)
        test.start()

        # Make prediction
        X_sample = np.random.randn(1, 10)
        variant, pred = test.predict(X_sample, sample_id="user_1")

        # Record outcome
        test.record_outcome("user_1", 1)

        # Check that outcome was recorded
        recorded_variant = test.variants[variant]
        assert len(recorded_variant.actuals) == 1

    def test_compute_metrics(self, simple_models: tuple) -> None:
        """Test metric computation."""
        model_a, model_b = simple_models
        test = ABTest("test")
        test.add_variant("control", model_a, 0.5)
        test.add_variant("treatment", model_b, 0.5)
        test.start()

        # Simulate some predictions
        X_test = np.random.randn(50, 10)
        y_test = np.random.randint(0, 2, 50)

        for i in range(50):
            variant, pred = test.predict(X_test[i : i + 1], sample_id=f"user_{i}")
            test.record_outcome(f"user_{i}", y_test[i])

        # Compute metrics
        metrics = test.compute_metrics()

        assert "control" in metrics or "treatment" in metrics
        for variant_metrics in metrics.values():
            if "accuracy" in variant_metrics:
                assert 0 <= variant_metrics["accuracy"] <= 1

    def test_get_summary(self, simple_models: tuple) -> None:
        """Test getting test summary."""
        model_a, model_b = simple_models
        test = ABTest("test")
        test.add_variant("control", model_a, 0.5)
        test.add_variant("treatment", model_b, 0.5)
        test.start()

        summary = test.get_summary()

        assert summary["name"] == "test"
        assert summary["status"] == "running"
        assert "control" in summary["variants"]


class TestABTestResult:
    """Tests for ABTestResult."""

    def test_result_str(self) -> None:
        """Test string representation."""
        result = ABTestResult(
            test_name="test",
            control_variant="A",
            treatment_variant="B",
            metric="accuracy",
            control_value=0.90,
            treatment_value=0.92,
            relative_improvement=0.022,
            p_value=0.03,
            is_significant=True,
            confidence_level=0.95,
            sample_sizes={"A": 1000, "B": 1000},
        )

        str_repr = str(result)

        assert "test" in str_repr
        assert "accuracy" in str_repr
        assert "0.90" in str_repr
        assert "YES" in str_repr


class TestRunOfflineABTest:
    """Tests for offline A/B testing."""

    def test_run_offline_test(self) -> None:
        """Test running an offline A/B test."""
        # Create models
        X_train = np.random.randn(200, 10)
        y_train = np.random.randint(0, 2, 200)
        X_test = np.random.randn(100, 10)
        y_test = np.random.randint(0, 2, 100)

        model_a = LogisticRegression(max_iter=200)
        model_b = LogisticRegression(max_iter=200, C=0.5)
        model_a.fit(X_train, y_train)
        model_b.fit(X_train, y_train)

        # Run test
        result = run_offline_ab_test(
            control_model=model_a,
            treatment_model=model_b,
            X_test=X_test,
            y_test=y_test,
        )

        assert isinstance(result, ABTestResult)
        assert 0 <= result.control_value <= 1
        assert 0 <= result.treatment_value <= 1
        assert 0 <= result.p_value <= 1


class TestComputeSampleSize:
    """Tests for sample size calculation."""

    def test_compute_sample_size(self) -> None:
        """Test sample size calculation."""
        n = compute_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.10,  # 10% relative
            significance_level=0.05,
            power=0.80,
        )

        assert n > 0
        assert isinstance(n, int)

    def test_larger_effect_smaller_sample(self) -> None:
        """Test that larger effect needs smaller sample."""
        n_small_effect = compute_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.05,
        )
        n_large_effect = compute_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.20,
        )

        assert n_large_effect < n_small_effect

    def test_higher_power_larger_sample(self) -> None:
        """Test that higher power needs larger sample."""
        n_low_power = compute_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.10,
            power=0.70,
        )
        n_high_power = compute_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.10,
            power=0.90,
        )

        assert n_high_power > n_low_power
