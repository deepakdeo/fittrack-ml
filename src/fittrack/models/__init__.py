"""ML model implementations."""

from fittrack.models.classical import (
    ClassicalModelTrainer,
    ModelConfig,
    TrainingResult,
    cross_validate_model,
    train_random_forest,
    train_with_tuning,
    train_xgboost,
)
from fittrack.models.data_loaders import (
    DataModule,
    HARDataset,
    TimeSeriesDataset,
    create_data_loaders,
    get_device,
)
from fittrack.models.deep_learning import (
    ActivityCNN,
    ActivityLSTM,
    HARClassifier,
    TrainingConfig,
    TrainingHistory,
    create_model,
    predict,
    train_model,
)
from fittrack.models.evaluation import (
    EvaluationMetrics,
    ModelEvaluator,
    compute_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
)

__all__ = [
    # Classical
    "ClassicalModelTrainer",
    "ModelConfig",
    "TrainingResult",
    "cross_validate_model",
    "train_random_forest",
    "train_with_tuning",
    "train_xgboost",
    # Data loaders
    "DataModule",
    "HARDataset",
    "TimeSeriesDataset",
    "create_data_loaders",
    "get_device",
    # Deep learning
    "ActivityCNN",
    "ActivityLSTM",
    "HARClassifier",
    "TrainingConfig",
    "TrainingHistory",
    "create_model",
    "predict",
    "train_model",
    # Evaluation
    "EvaluationMetrics",
    "ModelEvaluator",
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "plot_roc_curves",
]
