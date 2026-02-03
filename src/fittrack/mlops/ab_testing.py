"""A/B testing framework for model comparison.

This module provides utilities for running A/B tests between model variants,
including traffic splitting, metric collection, and statistical significance testing.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Status of an A/B test."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class Variant:
    """A/B test variant (model version).

    Attributes:
        name: Variant identifier (e.g., "control", "treatment").
        model: Model object or callable.
        traffic_weight: Proportion of traffic to send to this variant (0-1).
        predictions: Collected predictions.
        actuals: Actual labels (when available).
        metrics: Computed metrics.
    """

    name: str
    model: Any
    traffic_weight: float = 0.5
    predictions: list[int] = field(default_factory=list)
    actuals: list[int] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of predictions made."""
        return len(self.predictions)


@dataclass
class ABTestResult:
    """Results of an A/B test.

    Attributes:
        test_name: Name of the test.
        control_variant: Control variant name.
        treatment_variant: Treatment variant name.
        metric: Primary metric being compared.
        control_value: Metric value for control.
        treatment_value: Metric value for treatment.
        relative_improvement: (treatment - control) / control.
        p_value: Statistical significance p-value.
        is_significant: Whether the result is statistically significant.
        confidence_level: Confidence level used.
        sample_sizes: Number of samples per variant.
    """

    test_name: str
    control_variant: str
    treatment_variant: str
    metric: str
    control_value: float
    treatment_value: float
    relative_improvement: float
    p_value: float
    is_significant: bool
    confidence_level: float
    sample_sizes: dict[str, int]

    def __str__(self) -> str:
        """Return formatted string representation."""
        significant_str = "YES" if self.is_significant else "NO"
        return (
            f"A/B Test Results: {self.test_name}\n"
            f"{'=' * 50}\n"
            f"Metric: {self.metric}\n"
            f"Control ({self.control_variant}): {self.control_value:.4f}\n"
            f"Treatment ({self.treatment_variant}): {self.treatment_value:.4f}\n"
            f"Relative Improvement: {self.relative_improvement:+.2%}\n"
            f"P-value: {self.p_value:.4f}\n"
            f"Statistically Significant ({self.confidence_level:.0%}): {significant_str}\n"
            f"Sample sizes: {self.sample_sizes}"
        )


class TrafficSplitter:
    """Deterministic traffic splitting for A/B tests.

    Uses hashing for consistent assignment - the same input always
    goes to the same variant.

    Example:
        >>> splitter = TrafficSplitter({"control": 0.5, "treatment": 0.5})
        >>> variant = splitter.assign("user_123")
        >>> print(f"Assigned to: {variant}")
    """

    def __init__(self, weights: dict[str, float], seed: int = 42) -> None:
        """Initialize the splitter.

        Args:
            weights: Dictionary mapping variant names to traffic weights.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If weights don't sum to 1.0.
        """
        total = sum(weights.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Traffic weights must sum to 1.0, got {total}")

        self.weights = weights
        self.seed = seed
        self._variants = list(weights.keys())
        self._cumulative = np.cumsum(list(weights.values()))

    def assign(self, identifier: str | int) -> str:
        """Assign an identifier to a variant.

        Args:
            identifier: Unique identifier (user ID, session ID, etc.).

        Returns:
            Name of the assigned variant.
        """
        # Hash the identifier for deterministic assignment
        hash_value = hash((str(identifier), self.seed)) % 10000
        normalized = hash_value / 10000

        for i, threshold in enumerate(self._cumulative):
            if normalized < threshold:
                return self._variants[i]

        return self._variants[-1]

    def assign_batch(self, identifiers: list[str | int]) -> list[str]:
        """Assign multiple identifiers to variants.

        Args:
            identifiers: List of identifiers.

        Returns:
            List of variant assignments.
        """
        return [self.assign(id_) for id_ in identifiers]


class ABTest:
    """A/B test manager for comparing model variants.

    Example:
        >>> test = ABTest("rf-vs-xgb")
        >>> test.add_variant("control", rf_model, traffic_weight=0.5)
        >>> test.add_variant("treatment", xgb_model, traffic_weight=0.5)
        >>> test.start()
        >>>
        >>> # During inference
        >>> variant, prediction = test.predict(X_sample, sample_id="user_123")
        >>>
        >>> # When ground truth is available
        >>> test.record_outcome("user_123", actual_label)
        >>>
        >>> # Analyze results
        >>> result = test.analyze(metric="accuracy")
    """

    def __init__(
        self,
        name: str,
        confidence_level: float = 0.95,
    ) -> None:
        """Initialize the A/B test.

        Args:
            name: Test name/identifier.
            confidence_level: Confidence level for significance testing.
        """
        self.name = name
        self.confidence_level = confidence_level
        self.status = TestStatus.DRAFT
        self.variants: dict[str, Variant] = {}
        self.splitter: TrafficSplitter | None = None
        self.assignment_log: dict[str, str] = {}
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def add_variant(
        self,
        name: str,
        model: Any,
        traffic_weight: float = 0.5,
    ) -> None:
        """Add a variant to the test.

        Args:
            name: Variant name.
            model: Model object with predict() method.
            traffic_weight: Proportion of traffic for this variant.
        """
        if self.status != TestStatus.DRAFT:
            raise RuntimeError("Cannot add variants after test has started")

        self.variants[name] = Variant(
            name=name,
            model=model,
            traffic_weight=traffic_weight,
        )
        logger.info(f"Added variant '{name}' with weight {traffic_weight}")

    def start(self) -> None:
        """Start the A/B test."""
        if not self.variants:
            raise RuntimeError("No variants added to test")

        # Validate weights
        weights = {v.name: v.traffic_weight for v in self.variants.values()}
        self.splitter = TrafficSplitter(weights)

        self.status = TestStatus.RUNNING
        self.start_time = datetime.now()
        logger.info(f"Started A/B test '{self.name}'")

    def pause(self) -> None:
        """Pause the test."""
        if self.status != TestStatus.RUNNING:
            raise RuntimeError("Test is not running")
        self.status = TestStatus.PAUSED
        logger.info(f"Paused A/B test '{self.name}'")

    def resume(self) -> None:
        """Resume a paused test."""
        if self.status != TestStatus.PAUSED:
            raise RuntimeError("Test is not paused")
        self.status = TestStatus.RUNNING
        logger.info(f"Resumed A/B test '{self.name}'")

    def stop(self) -> None:
        """Stop the test."""
        self.status = TestStatus.STOPPED
        self.end_time = datetime.now()
        logger.info(f"Stopped A/B test '{self.name}'")

    def complete(self) -> None:
        """Mark the test as completed."""
        self.status = TestStatus.COMPLETED
        self.end_time = datetime.now()
        logger.info(f"Completed A/B test '{self.name}'")

    def predict(
        self,
        X: NDArray[np.floating],
        sample_id: str | int | None = None,
        variant_name: str | None = None,
    ) -> tuple[str, NDArray[np.integer]]:
        """Make a prediction using the assigned variant.

        Args:
            X: Input features (single sample or batch).
            sample_id: Identifier for traffic splitting.
            variant_name: Force a specific variant (overrides splitting).

        Returns:
            Tuple of (variant_name, predictions).
        """
        if self.status != TestStatus.RUNNING:
            raise RuntimeError(f"Test is not running (status: {self.status})")

        # Determine variant
        if variant_name is not None:
            assigned = variant_name
        elif sample_id is not None and self.splitter is not None:
            assigned = self.splitter.assign(sample_id)
            self.assignment_log[str(sample_id)] = assigned
        else:
            # Random assignment for this request
            assigned = np.random.choice(
                list(self.variants.keys()),
                p=[v.traffic_weight for v in self.variants.values()],
            )

        variant = self.variants[assigned]
        predictions = variant.model.predict(X)

        # Record predictions
        if X.ndim == 1:
            variant.predictions.append(int(predictions))
        else:
            variant.predictions.extend(predictions.tolist())

        return assigned, predictions

    def record_outcome(
        self,
        sample_id: str | int,
        actual: int,
    ) -> None:
        """Record the actual outcome for a prediction.

        Args:
            sample_id: Identifier from the prediction.
            actual: Actual/ground truth label.
        """
        variant_name = self.assignment_log.get(str(sample_id))
        if variant_name is None:
            logger.warning(f"No assignment found for sample_id: {sample_id}")
            return

        self.variants[variant_name].actuals.append(actual)

    def record_batch_outcomes(
        self,
        sample_ids: list[str | int],
        actuals: list[int],
    ) -> None:
        """Record outcomes for multiple predictions.

        Args:
            sample_ids: List of identifiers.
            actuals: List of actual labels.
        """
        for sample_id, actual in zip(sample_ids, actuals, strict=False):
            self.record_outcome(sample_id, actual)

    def compute_metrics(self) -> dict[str, dict[str, float]]:
        """Compute metrics for all variants.

        Returns:
            Dictionary mapping variant names to their metrics.
        """
        results = {}

        for name, variant in self.variants.items():
            if not variant.predictions or not variant.actuals:
                results[name] = {}
                continue

            # Align predictions and actuals (use minimum length)
            n = min(len(variant.predictions), len(variant.actuals))
            preds = np.array(variant.predictions[:n])
            actuals = np.array(variant.actuals[:n])

            # Compute metrics
            accuracy = np.mean(preds == actuals)
            variant.metrics = {
                "accuracy": float(accuracy),
                "n_samples": n,
                "n_correct": int(np.sum(preds == actuals)),
            }
            results[name] = variant.metrics

        return results

    def analyze(
        self,
        metric: str = "accuracy",
        control: str | None = None,
    ) -> ABTestResult:
        """Analyze the A/B test results.

        Args:
            metric: Metric to compare.
            control: Control variant name. Defaults to first variant.

        Returns:
            ABTestResult with statistical analysis.
        """
        # Compute metrics
        self.compute_metrics()

        # Determine control and treatment
        variant_names = list(self.variants.keys())
        if control is None:
            control = variant_names[0]

        treatments = [n for n in variant_names if n != control]
        if len(treatments) != 1:
            logger.warning("Multiple treatments; using first one")
        treatment = treatments[0]

        control_variant = self.variants[control]
        treatment_variant = self.variants[treatment]

        # Get metric values
        control_value = control_variant.metrics.get(metric, 0.0)
        treatment_value = treatment_variant.metrics.get(metric, 0.0)

        # Compute relative improvement
        if control_value > 0:
            relative_improvement = (treatment_value - control_value) / control_value
        else:
            relative_improvement = 0.0

        # Statistical significance test
        p_value = self._compute_significance(
            control_variant, treatment_variant, metric
        )

        alpha = 1 - self.confidence_level
        is_significant = p_value < alpha

        return ABTestResult(
            test_name=self.name,
            control_variant=control,
            treatment_variant=treatment,
            metric=metric,
            control_value=control_value,
            treatment_value=treatment_value,
            relative_improvement=relative_improvement,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            sample_sizes={
                control: control_variant.n_samples,
                treatment: treatment_variant.n_samples,
            },
        )

    def _compute_significance(
        self,
        control: Variant,
        treatment: Variant,
        metric: str,
    ) -> float:
        """Compute statistical significance using appropriate test.

        Args:
            control: Control variant.
            treatment: Treatment variant.
            metric: Metric being compared.

        Returns:
            P-value from statistical test.
        """
        if metric == "accuracy":
            # Use chi-square test for proportions
            return self._chi_square_test(control, treatment)
        else:
            # Use t-test for continuous metrics
            return self._t_test(control, treatment)

    def _chi_square_test(self, control: Variant, treatment: Variant) -> float:
        """Chi-square test for comparing proportions (accuracy).

        Args:
            control: Control variant.
            treatment: Treatment variant.

        Returns:
            P-value from chi-square test.
        """
        n_control = control.metrics.get("n_samples", 0)
        n_treatment = treatment.metrics.get("n_samples", 0)

        if n_control == 0 or n_treatment == 0:
            return 1.0

        correct_control = control.metrics.get("n_correct", 0)
        correct_treatment = treatment.metrics.get("n_correct", 0)

        # Contingency table
        # [[control_correct, control_incorrect], [treatment_correct, treatment_incorrect]]
        table = [
            [correct_control, n_control - correct_control],
            [correct_treatment, n_treatment - correct_treatment],
        ]

        # Chi-square test
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        return float(p_value)

    def _t_test(self, _control: Variant, _treatment: Variant) -> float:
        """Two-sample t-test for comparing means.

        Args:
            _control: Control variant.
            _treatment: Treatment variant.

        Returns:
            P-value from t-test.
        """
        # This would require storing individual metric values per prediction
        # For simplicity, return 1.0 (not significant) if we only have aggregates
        return 1.0

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the test status.

        Returns:
            Dictionary with test summary information.
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "variants": {
                name: {
                    "traffic_weight": v.traffic_weight,
                    "n_predictions": v.n_samples,
                    "metrics": v.metrics,
                }
                for name, v in self.variants.items()
            },
        }


def run_offline_ab_test(
    control_model: Any,
    treatment_model: Any,
    X_test: NDArray[np.floating],
    y_test: NDArray[np.integer],
    test_name: str = "offline-ab-test",
    control_name: str = "control",
    treatment_name: str = "treatment",
) -> ABTestResult:
    """Run an offline A/B test on held-out test data.

    This simulates an A/B test using existing test data, useful for
    comparing models before deploying to production.

    Args:
        control_model: Control model with predict() method.
        treatment_model: Treatment model with predict() method.
        X_test: Test features.
        y_test: Test labels.
        test_name: Name for the test.
        control_name: Name for control variant.
        treatment_name: Name for treatment variant.

    Returns:
        ABTestResult with comparison metrics.

    Example:
        >>> result = run_offline_ab_test(rf_model, xgb_model, X_test, y_test)
        >>> print(result)
    """
    # Get predictions from both models
    control_preds = control_model.predict(X_test)
    treatment_preds = treatment_model.predict(X_test)

    # Compute accuracies
    control_acc = np.mean(control_preds == y_test)
    treatment_acc = np.mean(treatment_preds == y_test)

    n_samples = len(y_test)
    control_correct = int(np.sum(control_preds == y_test))
    treatment_correct = int(np.sum(treatment_preds == y_test))

    # Chi-square test
    table = [
        [control_correct, n_samples - control_correct],
        [treatment_correct, n_samples - treatment_correct],
    ]
    _, p_value, _, _ = stats.chi2_contingency(table)

    # Relative improvement
    relative_improvement = (treatment_acc - control_acc) / control_acc if control_acc > 0 else 0.0

    return ABTestResult(
        test_name=test_name,
        control_variant=control_name,
        treatment_variant=treatment_name,
        metric="accuracy",
        control_value=float(control_acc),
        treatment_value=float(treatment_acc),
        relative_improvement=relative_improvement,
        p_value=float(p_value),
        is_significant=p_value < 0.05,
        confidence_level=0.95,
        sample_sizes={control_name: n_samples, treatment_name: n_samples},
    )


def compute_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    significance_level: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate required sample size for an A/B test.

    Args:
        baseline_rate: Expected conversion/accuracy rate for control (0-1).
        minimum_detectable_effect: Minimum relative effect size to detect.
        significance_level: Alpha (probability of false positive).
        power: 1 - beta (probability of detecting true effect).

    Returns:
        Required sample size per variant.

    Example:
        >>> n = compute_sample_size(
        ...     baseline_rate=0.90,
        ...     minimum_detectable_effect=0.05,  # 5% relative improvement
        ... )
        >>> print(f"Need {n} samples per variant")
    """
    from scipy.stats import norm

    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)

    # Pooled proportion
    p_pooled = (p1 + p2) / 2

    # Z-scores
    z_alpha = norm.ppf(1 - significance_level / 2)
    z_beta = norm.ppf(power)

    # Sample size formula for proportions
    numerator = (
        z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2

    n = int(np.ceil(numerator / denominator))

    logger.info(
        f"Required sample size: {n} per variant "
        f"(baseline={baseline_rate:.2%}, MDE={minimum_detectable_effect:.2%})"
    )

    return n
