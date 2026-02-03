"""Tests for classical machine learning models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from fittrack.models.classical import (
    ClassicalModelTrainer,
    ModelConfig,
    TrainingResult,
    cross_validate_model,
    get_rf_param_grid,
    get_top_features,
    get_xgb_param_grid,
    train_random_forest,
    train_xgboost,
    tune_hyperparameters,
)


@pytest.fixture
def sample_data() -> tuple:
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=4,
        random_state=42,
    )
    # Split into train/val
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]
    return X_train, y_train, X_val, y_val


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelConfig()
        assert config.n_estimators == 100
        assert config.max_depth is None
        assert config.random_state == 42
        assert config.n_jobs == -1

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ModelConfig(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
        )
        assert config.n_estimators == 200
        assert config.max_depth == 10
        assert config.learning_rate == 0.05


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_cv_properties(self) -> None:
        """Test cross-validation score properties."""
        result = TrainingResult(
            model=None,
            train_score=0.95,
            val_score=0.90,
            cv_scores=np.array([0.88, 0.90, 0.92, 0.89, 0.91]),
        )
        assert result.cv_mean == pytest.approx(0.90, rel=0.01)
        assert result.cv_std == pytest.approx(0.014, rel=0.1)

    def test_cv_properties_none(self) -> None:
        """Test CV properties when no CV was performed."""
        result = TrainingResult(
            model=None,
            train_score=0.95,
        )
        assert result.cv_mean is None
        assert result.cv_std is None


class TestRandomForest:
    """Tests for Random Forest training."""

    def test_train_random_forest(self, sample_data: tuple) -> None:
        """Test basic Random Forest training."""
        X_train, y_train, X_val, y_val = sample_data
        result = train_random_forest(X_train, y_train, X_val, y_val)

        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert 0.0 <= result.train_score <= 1.0
        assert 0.0 <= result.val_score <= 1.0
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 20

    def test_train_random_forest_with_config(self, sample_data: tuple) -> None:
        """Test Random Forest with custom config."""
        X_train, y_train, X_val, y_val = sample_data
        config = ModelConfig(n_estimators=50, max_depth=5)
        result = train_random_forest(X_train, y_train, X_val, y_val, config=config)

        assert result.model.n_estimators == 50
        assert result.model.max_depth == 5

    def test_train_random_forest_no_validation(self, sample_data: tuple) -> None:
        """Test training without validation data."""
        X_train, y_train, _, _ = sample_data
        result = train_random_forest(X_train, y_train)

        assert result.val_score is None
        assert result.train_score is not None


class TestXGBoost:
    """Tests for XGBoost training."""

    def test_train_xgboost(self, sample_data: tuple) -> None:
        """Test basic XGBoost training."""
        X_train, y_train, X_val, y_val = sample_data
        result = train_xgboost(X_train, y_train, X_val, y_val)

        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert 0.0 <= result.train_score <= 1.0
        assert 0.0 <= result.val_score <= 1.0

    def test_train_xgboost_with_config(self, sample_data: tuple) -> None:
        """Test XGBoost with custom config."""
        X_train, y_train, X_val, y_val = sample_data
        config = ModelConfig(n_estimators=50, max_depth=4, learning_rate=0.2)
        result = train_xgboost(X_train, y_train, X_val, y_val, config=config)

        assert result.model is not None


class TestCrossValidation:
    """Tests for cross-validation."""

    def test_cross_validate_model(self, sample_data: tuple) -> None:
        """Test cross-validation."""
        X_train, y_train, _, _ = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        scores = cross_validate_model(model, X_train, y_train, cv=3)

        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestHyperparameterTuning:
    """Tests for hyperparameter tuning."""

    def test_get_rf_param_grid(self) -> None:
        """Test Random Forest parameter grid."""
        grid = get_rf_param_grid("grid")
        assert "n_estimators" in grid
        assert "max_depth" in grid

        random_grid = get_rf_param_grid("random")
        assert "max_features" in random_grid

    def test_get_xgb_param_grid(self) -> None:
        """Test XGBoost parameter grid."""
        grid = get_xgb_param_grid("grid")
        assert "learning_rate" in grid
        assert "subsample" in grid

    def test_tune_hyperparameters(self, sample_data: tuple) -> None:
        """Test hyperparameter tuning (minimal search for speed)."""
        X_train, y_train, _, _ = sample_data
        model = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}

        best_model, best_params, best_score = tune_hyperparameters(
            model, X_train, y_train, param_grid, search_type="grid", cv=2
        )

        assert best_model is not None
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert 0.0 <= best_score <= 1.0


class TestFeatureImportance:
    """Tests for feature importance utilities."""

    def test_get_top_features(self) -> None:
        """Test getting top features."""
        importances = np.array([0.1, 0.3, 0.2, 0.4, 0.05])
        names = ["f1", "f2", "f3", "f4", "f5"]

        top_3 = get_top_features(importances, names, n_top=3)

        assert len(top_3) == 3
        assert top_3[0] == ("f4", 0.4)
        assert top_3[1] == ("f2", 0.3)
        assert top_3[2] == ("f3", 0.2)


class TestClassicalModelTrainer:
    """Tests for the unified trainer class."""

    def test_trainer_train_rf(self, sample_data: tuple) -> None:
        """Test training Random Forest via trainer."""
        X_train, y_train, X_val, y_val = sample_data
        trainer = ClassicalModelTrainer()

        result = trainer.train("random_forest", X_train, y_train, X_val, y_val, tune=False)

        assert "random_forest" in trainer.results
        assert result.model is not None

    def test_trainer_train_xgboost(self, sample_data: tuple) -> None:
        """Test training XGBoost via trainer."""
        X_train, y_train, X_val, y_val = sample_data
        trainer = ClassicalModelTrainer()

        result = trainer.train("xgboost", X_train, y_train, X_val, y_val, tune=False)

        assert "xgboost" in trainer.results
        assert result.model is not None

    def test_trainer_compare_models(self, sample_data: tuple) -> None:
        """Test model comparison."""
        X_train, y_train, X_val, y_val = sample_data
        trainer = ClassicalModelTrainer()

        trainer.train("random_forest", X_train, y_train, X_val, y_val)
        trainer.train("xgboost", X_train, y_train, X_val, y_val)

        comparison = trainer.compare_models()

        assert "random_forest" in comparison
        assert "xgboost" in comparison
        assert "train_score" in comparison["random_forest"]
        assert "val_score" in comparison["random_forest"]

    def test_trainer_get_best_model(self, sample_data: tuple) -> None:
        """Test getting best model."""
        X_train, y_train, X_val, y_val = sample_data
        trainer = ClassicalModelTrainer()

        trainer.train("random_forest", X_train, y_train, X_val, y_val)
        trainer.train("xgboost", X_train, y_train, X_val, y_val)

        best_name, best_model = trainer.get_best_model(metric="val_score")

        assert best_name in ["random_forest", "xgboost"]
        assert best_model is not None

    def test_trainer_invalid_model_type(self, sample_data: tuple) -> None:
        """Test that invalid model type raises error."""
        X_train, y_train, _, _ = sample_data
        trainer = ClassicalModelTrainer()

        with pytest.raises(ValueError, match="Unknown model type"):
            trainer.train("invalid_model", X_train, y_train)  # type: ignore
