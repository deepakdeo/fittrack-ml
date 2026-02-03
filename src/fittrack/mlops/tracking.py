"""MLflow tracking module for experiment management.

This module provides utilities for logging experiments, parameters,
metrics, and artifacts to MLflow.
"""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "mlruns"
DEFAULT_EXPERIMENT_NAME = "fittrack-har"


@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiment.

    Attributes:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: URI for the MLflow tracking server.
        artifact_location: Location for storing artifacts.
        tags: Default tags to add to all runs.
    """

    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    tracking_uri: str = DEFAULT_TRACKING_URI
    artifact_location: str | None = None
    tags: dict[str, str] | None = None


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> str:
    """Set up MLflow tracking.

    Args:
        experiment_name: Name of the experiment.
        tracking_uri: URI for tracking server. Defaults to local 'mlruns' directory.

    Returns:
        Experiment ID.

    Example:
        >>> experiment_id = setup_mlflow("my-experiment")
        >>> print(f"Experiment ID: {experiment_id}")
    """
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment '{experiment_name}' (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


@contextmanager
def start_run(
    run_name: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager for MLflow runs.

    Args:
        run_name: Name for the run.
        experiment_name: Name of the experiment.
        tags: Tags to add to the run.
        nested: Whether this is a nested run.

    Yields:
        Active MLflow run.

    Example:
        >>> with start_run("training-rf", tags={"model": "random_forest"}) as run:
        ...     mlflow.log_param("n_estimators", 100)
        ...     mlflow.log_metric("accuracy", 0.95)
    """
    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        logger.info(f"Started run: {run.info.run_name} (ID: {run.info.run_id})")
        yield run
        logger.info(f"Finished run: {run.info.run_name}")


def log_params(params: dict[str, Any]) -> None:
    """Log multiple parameters at once.

    Args:
        params: Dictionary of parameter names and values.

    Example:
        >>> log_params({"n_estimators": 100, "max_depth": 10})
    """
    mlflow.log_params(params)
    logger.debug(f"Logged {len(params)} parameters")


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log multiple metrics at once.

    Args:
        metrics: Dictionary of metric names and values.
        step: Optional step number for metric history.

    Example:
        >>> log_metrics({"accuracy": 0.95, "f1_score": 0.94})
    """
    mlflow.log_metrics(metrics, step=step)
    logger.debug(f"Logged {len(metrics)} metrics at step {step}")


def log_model(
    model: Any,
    model_type: str,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
    **kwargs: Any,
) -> None:
    """Log a trained model to MLflow.

    Args:
        model: Trained model object.
        model_type: Type of model ("sklearn", "pytorch", "xgboost").
        artifact_path: Path within the run's artifacts.
        registered_model_name: If provided, registers the model.
        **kwargs: Additional arguments for the model logger.

    Example:
        >>> log_model(rf_model, "sklearn", registered_model_name="har-classifier")
    """
    if model_type == "sklearn":
        mlflow.sklearn.log_model(
            model, artifact_path, registered_model_name=registered_model_name, **kwargs
        )
    elif model_type == "pytorch":
        mlflow.pytorch.log_model(
            model, artifact_path, registered_model_name=registered_model_name, **kwargs
        )
    elif model_type == "xgboost":
        mlflow.xgboost.log_model(
            model, artifact_path, registered_model_name=registered_model_name, **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Logged {model_type} model to '{artifact_path}'")


def log_artifact(local_path: str | Path, artifact_path: str | None = None) -> None:
    """Log a local file or directory as an artifact.

    Args:
        local_path: Path to the file or directory.
        artifact_path: Subdirectory in artifacts to place the file.

    Example:
        >>> log_artifact("confusion_matrix.png", "figures")
    """
    mlflow.log_artifact(str(local_path), artifact_path)
    logger.info(f"Logged artifact: {local_path}")


def log_figure(fig: Any, filename: str, artifact_path: str = "figures") -> None:
    """Log a matplotlib figure as an artifact.

    Args:
        fig: Matplotlib figure object.
        filename: Name for the saved figure file.
        artifact_path: Subdirectory in artifacts.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> log_figure(fig, "training_curve.png")
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(f.name, artifact_path)
        os.unlink(f.name)

    logger.info(f"Logged figure: {filename}")


def log_dict(data: dict[str, Any], filename: str, artifact_path: str | None = None) -> None:
    """Log a dictionary as a JSON artifact.

    Args:
        data: Dictionary to save.
        filename: Name for the JSON file.
        artifact_path: Subdirectory in artifacts.
    """
    mlflow.log_dict(data, f"{artifact_path}/{filename}" if artifact_path else filename)
    logger.info(f"Logged dict: {filename}")


class ExperimentTracker:
    """High-level experiment tracking interface.

    This class provides a clean API for tracking ML experiments,
    including parameters, metrics, models, and artifacts.

    Example:
        >>> tracker = ExperimentTracker("my-experiment")
        >>> with tracker.start_run("rf-baseline") as run:
        ...     tracker.log_params({"n_estimators": 100})
        ...     # Train model...
        ...     tracker.log_metrics({"accuracy": 0.95})
        ...     tracker.log_model(model, "sklearn")
    """

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        tracking_uri: str | None = None,
    ) -> None:
        """Initialize the tracker.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: URI for tracking server.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.experiment_id = setup_mlflow(experiment_name, tracking_uri)
        self.client = MlflowClient()
        self._active_run: mlflow.ActiveRun | None = None

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Start a new tracking run.

        Args:
            run_name: Name for the run.
            tags: Tags to add to the run.

        Yields:
            Active MLflow run.
        """
        with start_run(run_name, self.experiment_name, tags) as run:
            self._active_run = run
            yield run
            self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for the current run."""
        log_params(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics for the current run."""
        log_metrics(metrics, step)

    def log_model(
        self,
        model: Any,
        model_type: str,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Log a model for the current run."""
        log_model(model, model_type, artifact_path, registered_model_name)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Log an artifact for the current run."""
        log_artifact(local_path, artifact_path)

    def log_figure(
        self,
        fig: Any,
        filename: str,
        artifact_path: str = "figures",
    ) -> None:
        """Log a figure for the current run."""
        log_figure(fig, filename, artifact_path)

    def get_best_run(
        self,
        metric: str = "accuracy",
        ascending: bool = False,
    ) -> dict[str, Any] | None:
        """Get the best run based on a metric.

        Args:
            metric: Metric to compare.
            ascending: If True, lower is better.

        Returns:
            Dictionary with run info, or None if no runs found.
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if not runs:
            return None

        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "run_name": best_run.info.run_name,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
        }

    def get_run_history(
        self,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Get history of all runs.

        Args:
            max_results: Maximum number of runs to return.

        Returns:
            List of run dictionaries.
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=max_results,
        )

        return [
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }
            for run in runs
        ]

    def compare_runs(
        self,
        metrics: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compare metrics across all runs.

        Args:
            metrics: List of metrics to compare. None for all.

        Returns:
            Dictionary mapping run names to their metrics.
        """
        runs = self.get_run_history()
        comparison = {}

        for run in runs:
            run_name = run["run_name"] or run["run_id"][:8]
            if metrics:
                comparison[run_name] = {
                    m: run["metrics"].get(m, None) for m in metrics
                }
            else:
                comparison[run_name] = run["metrics"]

        return comparison


def log_training_run(
    model: Any,
    model_type: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    artifacts: dict[str, str | Path] | None = None,
    register_model: bool = False,
    model_name: str | None = None,
) -> str:
    """Convenience function to log a complete training run.

    Args:
        model: Trained model.
        model_type: Type of model ("sklearn", "pytorch", "xgboost").
        params: Training parameters.
        metrics: Evaluation metrics.
        experiment_name: MLflow experiment name.
        run_name: Name for this run.
        artifacts: Dictionary of artifact names to paths.
        register_model: Whether to register the model.
        model_name: Name for registered model.

    Returns:
        Run ID.

    Example:
        >>> run_id = log_training_run(
        ...     model=rf_model,
        ...     model_type="sklearn",
        ...     params={"n_estimators": 100},
        ...     metrics={"accuracy": 0.95, "f1": 0.94},
        ...     run_name="rf-baseline",
        ...     register_model=True,
        ...     model_name="har-classifier",
        ... )
    """
    with start_run(run_name, experiment_name) as run:
        # Log parameters
        log_params(params)

        # Log metrics
        log_metrics(metrics)

        # Log model
        registered_name = model_name if register_model else None
        log_model(model, model_type, registered_model_name=registered_name)

        # Log artifacts
        if artifacts:
            for name, path in artifacts.items():
                log_artifact(path, name)

        return run.info.run_id
