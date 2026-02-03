"""MLOps utilities for experiment tracking and model registry."""

from fittrack.mlops.ab_testing import (
    ABTest,
    ABTestResult,
    TrafficSplitter,
    compute_sample_size,
    run_offline_ab_test,
)
from fittrack.mlops.registry import (
    ModelRegistry,
    ModelStage,
    ModelVersion,
    get_best_model_version,
    promote_best_to_production,
)
from fittrack.mlops.tracking import (
    ExperimentConfig,
    ExperimentTracker,
    log_artifact,
    log_metrics,
    log_model,
    log_params,
    log_training_run,
    setup_mlflow,
    start_run,
)

__all__ = [
    # A/B Testing
    "ABTest",
    "ABTestResult",
    "TrafficSplitter",
    "compute_sample_size",
    "run_offline_ab_test",
    # Registry
    "ModelRegistry",
    "ModelStage",
    "ModelVersion",
    "get_best_model_version",
    "promote_best_to_production",
    # Tracking
    "ExperimentConfig",
    "ExperimentTracker",
    "log_artifact",
    "log_metrics",
    "log_model",
    "log_params",
    "log_training_run",
    "setup_mlflow",
    "start_run",
]
