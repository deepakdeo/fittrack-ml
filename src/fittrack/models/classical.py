"""Classical machine learning models for activity recognition.

This module provides Random Forest and XGBoost classifiers with
hyperparameter tuning and cross-validation utilities.
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training.

    Attributes:
        n_estimators: Number of trees (RF) or boosting rounds (XGBoost).
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples to split (RF only).
        learning_rate: Learning rate (XGBoost only).
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 for all cores).
    """

    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    learning_rate: float = 0.1
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class TrainingResult:
    """Container for training results.

    Attributes:
        model: Trained model instance.
        train_score: Training accuracy.
        val_score: Validation accuracy (if provided).
        cv_scores: Cross-validation scores (if performed).
        best_params: Best hyperparameters (if tuning was performed).
        feature_importances: Feature importance values.
    """

    model: Any
    train_score: float
    val_score: float | None = None
    cv_scores: NDArray[np.floating] | None = None
    best_params: dict[str, Any] | None = None
    feature_importances: NDArray[np.floating] | None = None

    @property
    def cv_mean(self) -> float | None:
        """Return mean cross-validation score."""
        if self.cv_scores is not None:
            return float(np.mean(self.cv_scores))
        return None

    @property
    def cv_std(self) -> float | None:
        """Return std of cross-validation scores."""
        if self.cv_scores is not None:
            return float(np.std(self.cv_scores))
        return None


def train_random_forest(
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    X_val: NDArray[np.floating] | None = None,
    y_val: NDArray[np.integer] | None = None,
    config: ModelConfig | None = None,
    sample_weights: NDArray[np.floating] | None = None,
) -> TrainingResult:
    """Train a Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        config: Model configuration. Uses defaults if None.
        sample_weights: Per-sample weights for handling class imbalance.

    Returns:
        TrainingResult containing the trained model and metrics.

    Example:
        >>> result = train_random_forest(X_train, y_train, X_val, y_val)
        >>> print(f"Validation accuracy: {result.val_score:.3f}")
    """
    if config is None:
        config = ModelConfig()

    logger.info(
        f"Training Random Forest with n_estimators={config.n_estimators}, "
        f"max_depth={config.max_depth}"
    )

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        class_weight="balanced",
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    train_score = float(model.score(X_train, y_train))
    val_score = None
    if X_val is not None and y_val is not None:
        val_score = float(model.score(X_val, y_val))

    logger.info(f"Train accuracy: {train_score:.4f}")
    if val_score is not None:
        logger.info(f"Validation accuracy: {val_score:.4f}")

    return TrainingResult(
        model=model,
        train_score=train_score,
        val_score=val_score,
        feature_importances=model.feature_importances_,
    )


def train_xgboost(
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    X_val: NDArray[np.floating] | None = None,
    y_val: NDArray[np.integer] | None = None,
    config: ModelConfig | None = None,
    sample_weights: NDArray[np.floating] | None = None,
) -> TrainingResult:
    """Train an XGBoost classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        config: Model configuration. Uses defaults if None.
        sample_weights: Per-sample weights for handling class imbalance.

    Returns:
        TrainingResult containing the trained model and metrics.

    Example:
        >>> result = train_xgboost(X_train, y_train, X_val, y_val)
        >>> print(f"Validation accuracy: {result.val_score:.3f}")
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as err:
        raise ImportError("XGBoost is required. Install with: pip install xgboost") from err

    if config is None:
        config = ModelConfig()

    logger.info(
        f"Training XGBoost with n_estimators={config.n_estimators}, "
        f"max_depth={config.max_depth or 6}, learning_rate={config.learning_rate}"
    )

    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth or 6,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )

    # Prepare eval set if validation data provided
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=False,
    )

    train_score = float(model.score(X_train, y_train))
    val_score = None
    if X_val is not None and y_val is not None:
        val_score = float(model.score(X_val, y_val))

    logger.info(f"Train accuracy: {train_score:.4f}")
    if val_score is not None:
        logger.info(f"Validation accuracy: {val_score:.4f}")

    return TrainingResult(
        model=model,
        train_score=train_score,
        val_score=val_score,
        feature_importances=model.feature_importances_,
    )


def cross_validate_model(
    model: Any,
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    cv: int = 5,
    scoring: str = "accuracy",
) -> NDArray[np.floating]:
    """Perform cross-validation on a model.

    Args:
        model: Scikit-learn compatible model.
        X: Features.
        y: Labels.
        cv: Number of cross-validation folds.
        scoring: Scoring metric.

    Returns:
        Array of scores for each fold.

    Example:
        >>> rf = RandomForestClassifier()
        >>> scores = cross_validate_model(rf, X, y, cv=5)
        >>> print(f"CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    """
    logger.info(f"Running {cv}-fold cross-validation with {scoring} metric")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    logger.info(f"CV scores: {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores


def get_rf_param_grid(search_type: Literal["grid", "random"] = "grid") -> dict[str, Any]:
    """Get hyperparameter search space for Random Forest.

    Args:
        search_type: Type of search ("grid" for exhaustive, "random" for sampling).

    Returns:
        Dictionary of hyperparameter ranges.
    """
    if search_type == "grid":
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    else:
        return {
            "n_estimators": [50, 100, 150, 200, 300],
            "max_depth": [None, 5, 10, 15, 20, 30],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        }


def get_xgb_param_grid(search_type: Literal["grid", "random"] = "grid") -> dict[str, Any]:
    """Get hyperparameter search space for XGBoost.

    Args:
        search_type: Type of search ("grid" for exhaustive, "random" for sampling).

    Returns:
        Dictionary of hyperparameter ranges.
    """
    if search_type == "grid":
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        }
    else:
        return {
            "n_estimators": [50, 100, 150, 200, 300],
            "max_depth": [3, 4, 5, 6, 7, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
        }


def tune_hyperparameters(
    model: Any,
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    param_grid: dict[str, Any],
    search_type: Literal["grid", "random"] = "grid",
    cv: int = 3,
    n_iter: int = 50,
    scoring: str = "accuracy",
) -> tuple[Any, dict[str, Any], float]:
    """Tune hyperparameters using grid or random search.

    Args:
        model: Base model to tune.
        X_train: Training features.
        y_train: Training labels.
        param_grid: Hyperparameter search space.
        search_type: Type of search ("grid" or "random").
        cv: Number of cross-validation folds.
        n_iter: Number of iterations for random search.
        scoring: Scoring metric.

    Returns:
        Tuple of (best_model, best_params, best_score).

    Example:
        >>> rf = RandomForestClassifier()
        >>> param_grid = get_rf_param_grid("random")
        >>> best_model, best_params, best_score = tune_hyperparameters(
        ...     rf, X_train, y_train, param_grid, search_type="random"
        ... )
    """
    logger.info(
        f"Starting {search_type} search with {cv}-fold CV, "
        f"scoring={scoring}"
    )

    if search_type == "grid":
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )
    else:
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )

    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_


def train_with_tuning(
    model_type: Literal["random_forest", "xgboost"],
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    X_val: NDArray[np.floating] | None = None,
    y_val: NDArray[np.integer] | None = None,
    search_type: Literal["grid", "random"] = "random",
    cv: int = 3,
    n_iter: int = 30,
) -> TrainingResult:
    """Train a model with hyperparameter tuning.

    Args:
        model_type: Type of model ("random_forest" or "xgboost").
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        search_type: Type of hyperparameter search.
        cv: Number of CV folds for tuning.
        n_iter: Number of iterations for random search.

    Returns:
        TrainingResult with best model and metrics.

    Example:
        >>> result = train_with_tuning(
        ...     "random_forest", X_train, y_train, X_val, y_val,
        ...     search_type="random", n_iter=20
        ... )
        >>> print(f"Best params: {result.best_params}")
    """
    if model_type == "random_forest":
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
        param_grid = get_rf_param_grid(search_type)
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        base_model = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        param_grid = get_xgb_param_grid(search_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    best_model, best_params, best_cv_score = tune_hyperparameters(
        base_model,
        X_train,
        y_train,
        param_grid,
        search_type=search_type,
        cv=cv,
        n_iter=n_iter,
    )

    train_score = float(best_model.score(X_train, y_train))
    val_score = None
    if X_val is not None and y_val is not None:
        val_score = float(best_model.score(X_val, y_val))

    return TrainingResult(
        model=best_model,
        train_score=train_score,
        val_score=val_score,
        best_params=best_params,
        feature_importances=best_model.feature_importances_,
    )


def get_top_features(
    feature_importances: NDArray[np.floating],
    feature_names: list[str],
    n_top: int = 20,
) -> list[tuple[str, float]]:
    """Get top N most important features.

    Args:
        feature_importances: Array of feature importance values.
        feature_names: List of feature names.
        n_top: Number of top features to return.

    Returns:
        List of (feature_name, importance) tuples sorted by importance.
    """
    indices = np.argsort(feature_importances)[::-1][:n_top]
    return [(feature_names[i], float(feature_importances[i])) for i in indices]


class ClassicalModelTrainer:
    """Unified trainer for classical ML models.

    Example:
        >>> trainer = ClassicalModelTrainer()
        >>> rf_result = trainer.train("random_forest", X_train, y_train, X_val, y_val)
        >>> xgb_result = trainer.train("xgboost", X_train, y_train, X_val, y_val)
        >>> comparison = trainer.compare_models()
    """

    def __init__(self) -> None:
        """Initialize the trainer."""
        self.results: dict[str, TrainingResult] = {}

    def train(
        self,
        model_type: Literal["random_forest", "xgboost"],
        X_train: NDArray[np.floating],
        y_train: NDArray[np.integer],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.integer] | None = None,
        tune: bool = False,
        config: ModelConfig | None = None,
    ) -> TrainingResult:
        """Train a model and store the result.

        Args:
            model_type: Type of model to train.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            tune: Whether to perform hyperparameter tuning.
            config: Model configuration (ignored if tune=True).

        Returns:
            TrainingResult for the trained model.
        """
        if tune:
            result = train_with_tuning(
                model_type, X_train, y_train, X_val, y_val
            )
        else:
            if model_type == "random_forest":
                result = train_random_forest(
                    X_train, y_train, X_val, y_val, config
                )
            elif model_type == "xgboost":
                result = train_xgboost(
                    X_train, y_train, X_val, y_val, config
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        self.results[model_type] = result
        return result

    def compare_models(self) -> dict[str, dict[str, float | None]]:
        """Compare all trained models.

        Returns:
            Dictionary with model comparison metrics.
        """
        comparison = {}
        for name, result in self.results.items():
            comparison[name] = {
                "train_score": result.train_score,
                "val_score": result.val_score,
                "cv_mean": result.cv_mean,
            }
        return comparison

    def get_best_model(
        self, metric: Literal["val_score", "train_score"] = "val_score"
    ) -> tuple[str, Any]:
        """Get the best performing model.

        Args:
            metric: Metric to use for comparison.

        Returns:
            Tuple of (model_name, model_instance).
        """
        best_name = None
        best_score = -float("inf")

        for name, result in self.results.items():
            score = getattr(result, metric)
            if score is not None and score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            raise ValueError("No models have been trained yet")

        return best_name, self.results[best_name].model
