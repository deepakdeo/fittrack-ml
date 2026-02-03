"""Model registry module for versioning and lifecycle management.

This module provides utilities for managing model versions,
promoting models through staging/production, and loading deployed models.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import mlflow
from mlflow.tracking import MlflowClient

from fittrack.mlops.tracking import setup_mlflow

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model lifecycle stages."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class ModelVersion:
    """Container for model version information.

    Attributes:
        name: Registered model name.
        version: Version number.
        stage: Current lifecycle stage.
        run_id: MLflow run ID that created this version.
        creation_time: When the version was created.
        description: Version description.
        tags: Version tags.
    """

    name: str
    version: int
    stage: str
    run_id: str
    creation_time: datetime
    description: str | None = None
    tags: dict[str, str] | None = None


class ModelRegistry:
    """Interface for MLflow Model Registry.

    Provides functionality for:
    - Registering models
    - Managing model versions
    - Promoting models through lifecycle stages
    - Loading production models

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register_model("runs:/abc123/model", "har-classifier")
        >>> registry.transition_stage("har-classifier", 1, "Production")
        >>> model = registry.load_production_model("har-classifier")
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        """Initialize the registry.

        Args:
            tracking_uri: MLflow tracking server URI.
        """
        setup_mlflow(tracking_uri=tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        name: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> ModelVersion:
        """Register a model from a run.

        Args:
            model_uri: URI of the model (e.g., "runs:/run_id/model").
            name: Name for the registered model.
            description: Model description.
            tags: Tags to add to the model version.

        Returns:
            ModelVersion with registration details.

        Example:
            >>> version = registry.register_model(
            ...     "runs:/abc123/model",
            ...     "har-classifier",
            ...     description="Random Forest baseline",
            ... )
        """
        # Create or get registered model
        try:
            self.client.get_registered_model(name)
            logger.info(f"Using existing registered model: {name}")
        except mlflow.exceptions.RestException:
            self.client.create_registered_model(name, description=description)
            logger.info(f"Created registered model: {name}")

        # Register the version
        mv = mlflow.register_model(model_uri, name)

        # Add description and tags
        if description:
            self.client.update_model_version(name, mv.version, description=description)
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(name, mv.version, key, value)

        logger.info(f"Registered model '{name}' version {mv.version}")

        return ModelVersion(
            name=name,
            version=int(mv.version),
            stage=mv.current_stage,
            run_id=mv.run_id,
            creation_time=datetime.fromtimestamp(mv.creation_timestamp / 1000),
            description=description,
            tags=tags,
        )

    def transition_stage(
        self,
        name: str,
        version: int,
        stage: Literal["Staging", "Production", "Archived"],
        archive_existing: bool = True,
    ) -> None:
        """Transition a model version to a new stage.

        Args:
            name: Registered model name.
            version: Version number to transition.
            stage: Target stage.
            archive_existing: If True, archive existing models in the target stage.

        Example:
            >>> registry.transition_stage("har-classifier", 1, "Production")
        """
        self.client.transition_model_version_stage(
            name=name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Transitioned {name} v{version} to {stage}")

    def get_latest_version(
        self,
        name: str,
        stage: str | None = None,
    ) -> ModelVersion | None:
        """Get the latest version of a model.

        Args:
            name: Registered model name.
            stage: Filter by stage (optional).

        Returns:
            Latest ModelVersion or None if not found.
        """
        stages = [stage] if stage else None
        versions = self.client.get_latest_versions(name, stages=stages)

        if not versions:
            return None

        mv = versions[0]
        return ModelVersion(
            name=name,
            version=int(mv.version),
            stage=mv.current_stage,
            run_id=mv.run_id,
            creation_time=datetime.fromtimestamp(mv.creation_timestamp / 1000),
            description=mv.description,
        )

    def get_production_version(self, name: str) -> ModelVersion | None:
        """Get the production version of a model.

        Args:
            name: Registered model name.

        Returns:
            Production ModelVersion or None if not found.
        """
        return self.get_latest_version(name, stage="Production")

    def get_all_versions(self, name: str) -> list[ModelVersion]:
        """Get all versions of a registered model.

        Args:
            name: Registered model name.

        Returns:
            List of ModelVersion objects.
        """
        versions = self.client.search_model_versions(f"name='{name}'")

        return [
            ModelVersion(
                name=name,
                version=int(mv.version),
                stage=mv.current_stage,
                run_id=mv.run_id,
                creation_time=datetime.fromtimestamp(mv.creation_timestamp / 1000),
                description=mv.description,
            )
            for mv in versions
        ]

    def load_model(
        self,
        name: str,
        version: int | None = None,
        stage: str | None = None,
    ) -> Any:
        """Load a model from the registry.

        Args:
            name: Registered model name.
            version: Specific version to load.
            stage: Load from a specific stage (e.g., "Production").

        Returns:
            Loaded model object.

        Raises:
            ValueError: If neither version nor stage is specified.

        Example:
            >>> model = registry.load_model("har-classifier", stage="Production")
            >>> predictions = model.predict(X_test)
        """
        if version is not None:
            model_uri = f"models:/{name}/{version}"
        elif stage is not None:
            model_uri = f"models:/{name}/{stage}"
        else:
            raise ValueError("Must specify either version or stage")

        logger.info(f"Loading model from {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)

    def load_production_model(self, name: str) -> Any:
        """Load the production version of a model.

        Args:
            name: Registered model name.

        Returns:
            Loaded production model.
        """
        return self.load_model(name, stage="Production")

    def load_sklearn_model(
        self,
        name: str,
        version: int | None = None,
        stage: str | None = None,
    ) -> Any:
        """Load a sklearn model from the registry.

        Args:
            name: Registered model name.
            version: Specific version to load.
            stage: Load from a specific stage.

        Returns:
            Loaded sklearn model.
        """
        if version is not None:
            model_uri = f"models:/{name}/{version}"
        elif stage is not None:
            model_uri = f"models:/{name}/{stage}"
        else:
            raise ValueError("Must specify either version or stage")

        return mlflow.sklearn.load_model(model_uri)

    def load_pytorch_model(
        self,
        name: str,
        version: int | None = None,
        stage: str | None = None,
    ) -> Any:
        """Load a PyTorch model from the registry.

        Args:
            name: Registered model name.
            version: Specific version to load.
            stage: Load from a specific stage.

        Returns:
            Loaded PyTorch model.
        """
        if version is not None:
            model_uri = f"models:/{name}/{version}"
        elif stage is not None:
            model_uri = f"models:/{name}/{stage}"
        else:
            raise ValueError("Must specify either version or stage")

        return mlflow.pytorch.load_model(model_uri)

    def delete_version(self, name: str, version: int) -> None:
        """Delete a specific model version.

        Args:
            name: Registered model name.
            version: Version to delete.
        """
        self.client.delete_model_version(name, str(version))
        logger.info(f"Deleted {name} v{version}")

    def delete_model(self, name: str) -> None:
        """Delete a registered model and all its versions.

        Args:
            name: Registered model name.
        """
        self.client.delete_registered_model(name)
        logger.info(f"Deleted registered model: {name}")

    def set_model_description(self, name: str, description: str) -> None:
        """Set description for a registered model.

        Args:
            name: Registered model name.
            description: New description.
        """
        self.client.update_registered_model(name, description=description)

    def set_version_tag(
        self,
        name: str,
        version: int,
        key: str,
        value: str,
    ) -> None:
        """Set a tag on a model version.

        Args:
            name: Registered model name.
            version: Version number.
            key: Tag key.
            value: Tag value.
        """
        self.client.set_model_version_tag(name, str(version), key, value)

    def compare_versions(
        self,
        name: str,
        versions: list[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Compare multiple model versions.

        Args:
            name: Registered model name.
            versions: Specific versions to compare. None for all.

        Returns:
            Dictionary mapping version numbers to their metadata.
        """
        all_versions = self.get_all_versions(name)

        if versions:
            all_versions = [v for v in all_versions if v.version in versions]

        comparison = {}
        for mv in all_versions:
            # Get run metrics
            run = self.client.get_run(mv.run_id)
            comparison[mv.version] = {
                "stage": mv.stage,
                "created": mv.creation_time.isoformat(),
                "run_id": mv.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }

        return comparison


def get_best_model_version(
    name: str,
    metric: str = "accuracy",
    ascending: bool = False,
) -> tuple[int, float]:
    """Find the best model version based on a metric.

    Args:
        name: Registered model name.
        metric: Metric to compare.
        ascending: If True, lower is better.

    Returns:
        Tuple of (version, metric_value).
    """
    registry = ModelRegistry()
    comparison = registry.compare_versions(name)

    best_version = None
    best_value = float("inf") if ascending else float("-inf")

    for version, data in comparison.items():
        value = data["metrics"].get(metric)
        if value is not None and (
            (ascending and value < best_value) or (not ascending and value > best_value)
        ):
            best_version = version
            best_value = value

    if best_version is None:
        raise ValueError(f"No versions found with metric '{metric}'")

    return best_version, best_value


def promote_best_to_production(
    name: str,
    metric: str = "accuracy",
    ascending: bool = False,
) -> int:
    """Promote the best model version to production.

    Args:
        name: Registered model name.
        metric: Metric to compare.
        ascending: If True, lower is better.

    Returns:
        Version number that was promoted.

    Example:
        >>> version = promote_best_to_production("har-classifier", metric="f1_score")
        >>> print(f"Promoted version {version} to Production")
    """
    best_version, best_value = get_best_model_version(name, metric, ascending)

    registry = ModelRegistry()
    registry.transition_stage(name, best_version, "Production")

    logger.info(f"Promoted {name} v{best_version} to Production ({metric}={best_value:.4f})")

    return best_version
