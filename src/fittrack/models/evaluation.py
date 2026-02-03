"""Model evaluation module for activity recognition.

This module provides metrics computation, confusion matrix visualization,
ROC curves, and comprehensive model evaluation utilities.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics.

    Attributes:
        accuracy: Overall accuracy.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        f1_macro: Macro-averaged F1 score.
        precision_weighted: Weighted precision.
        recall_weighted: Weighted recall.
        f1_weighted: Weighted F1 score.
        per_class_precision: Precision per class.
        per_class_recall: Recall per class.
        per_class_f1: F1 score per class.
        confusion_matrix: Confusion matrix.
        roc_auc: ROC AUC score (if probabilities available).
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class_precision: NDArray[np.floating]
    per_class_recall: NDArray[np.floating]
    per_class_f1: NDArray[np.floating]
    confusion_matrix: NDArray[np.integer]
    roc_auc: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert scalar metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "roc_auc": self.roc_auc if self.roc_auc is not None else 0.0,
        }

    def __str__(self) -> str:
        """Return formatted string representation."""
        lines = [
            "Evaluation Metrics:",
            f"  Accuracy:           {self.accuracy:.4f}",
            f"  Precision (macro):  {self.precision_macro:.4f}",
            f"  Recall (macro):     {self.recall_macro:.4f}",
            f"  F1 Score (macro):   {self.f1_macro:.4f}",
            f"  F1 Score (weighted): {self.f1_weighted:.4f}",
        ]
        if self.roc_auc is not None:
            lines.append(f"  ROC AUC:            {self.roc_auc:.4f}")
        return "\n".join(lines)


def compute_metrics(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    y_proba: NDArray[np.floating] | None = None,
    class_names: list[str] | None = None,
) -> EvaluationMetrics:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (for ROC AUC).
        class_names: Names of classes for logging.

    Returns:
        EvaluationMetrics containing all computed metrics.

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 2, 0, 2, 2])
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics.accuracy:.3f}")
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Macro and weighted averages
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC (requires probabilities and multi-class handling)
    roc_auc = None
    if y_proba is not None:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            # Multi-class: one-vs-rest
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            roc_auc = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")

    logger.info(f"Computed metrics - Accuracy: {accuracy:.4f}, F1 (macro): {f1_macro:.4f}")

    return EvaluationMetrics(
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        f1_weighted=f1_weighted,
        per_class_precision=precision_per_class,
        per_class_recall=recall_per_class,
        per_class_f1=f1_per_class,
        confusion_matrix=cm,
        roc_auc=roc_auc,
    )


def plot_confusion_matrix(
    cm: NDArray[np.integer],
    class_names: list[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (10, 8),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array.
        class_names: Names of classes.
        normalize: Whether to normalize values (show percentages).
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure (optional).

    Returns:
        Matplotlib Figure object.

    Example:
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> fig = plot_confusion_matrix(cm, ["Walking", "Sitting", "Running"])
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = None

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")

    return fig


def plot_roc_curves(
    y_true: NDArray[np.integer],
    y_proba: NDArray[np.floating],
    class_names: list[str],
    title: str = "ROC Curves",
    figsize: tuple[int, int] = (10, 8),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot ROC curves for multi-class classification.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities (n_samples, n_classes).
        class_names: Names of classes.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure (optional).

    Returns:
        Matplotlib Figure object.

    Example:
        >>> y_proba = model.predict_proba(X_test)
        >>> fig = plot_roc_curves(y_test, y_proba, class_names)
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_classes))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{class_name} (AUC = {auc:.3f})")

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ROC curves to {save_path}")

    return fig


def plot_precision_recall_per_class(
    metrics: EvaluationMetrics,
    class_names: list[str],
    title: str = "Per-Class Metrics",
    figsize: tuple[int, int] = (12, 6),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot precision, recall, and F1 for each class.

    Args:
        metrics: EvaluationMetrics object.
        class_names: Names of classes.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure (optional).

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(class_names))
    width = 0.25

    ax.bar(x - width, metrics.per_class_precision, width, label="Precision", color="steelblue")
    ax.bar(x, metrics.per_class_recall, width, label="Recall", color="coral")
    ax.bar(x + width, metrics.per_class_f1, width, label="F1 Score", color="forestgreen")

    ax.set_xlabel("Activity Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for i, (p, r, f) in enumerate(
        zip(metrics.per_class_precision, metrics.per_class_recall, metrics.per_class_f1)
    ):
        ax.text(i - width, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + width, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved per-class metrics to {save_path}")

    return fig


def plot_model_comparison(
    model_metrics: dict[str, EvaluationMetrics],
    metric_names: list[str] | None = None,
    title: str = "Model Comparison",
    figsize: tuple[int, int] = (10, 6),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot comparison of multiple models.

    Args:
        model_metrics: Dictionary mapping model names to their metrics.
        metric_names: Which metrics to compare. Defaults to common metrics.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure (optional).

    Returns:
        Matplotlib Figure object.

    Example:
        >>> metrics = {"RF": rf_metrics, "XGBoost": xgb_metrics}
        >>> fig = plot_model_comparison(metrics)
    """
    if metric_names is None:
        metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(model_metrics.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)

    colors = plt.cm.get_cmap("Set2")(np.linspace(0, 1, len(model_names)))

    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        values = [getattr(metrics, m) for m in metric_names]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names])
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved model comparison to {save_path}")

    return fig


def print_classification_report(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    class_names: list[str],
) -> str:
    """Print and return sklearn classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Names of classes.

    Returns:
        Classification report string.
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    return report


class ModelEvaluator:
    """Comprehensive model evaluation utility.

    Example:
        >>> evaluator = ModelEvaluator(class_names=["Walking", "Sitting", "Running"])
        >>> metrics = evaluator.evaluate(model, X_test, y_test)
        >>> evaluator.plot_all(save_dir="./figures")
    """

    def __init__(
        self,
        class_names: list[str],
        figures_dir: Path | str | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            class_names: Names of activity classes.
            figures_dir: Directory to save figures.
        """
        self.class_names = class_names
        self.figures_dir = Path(figures_dir) if figures_dir else None
        self.results: dict[str, EvaluationMetrics] = {}
        self._predictions: dict[str, tuple[NDArray, NDArray, NDArray | None]] = {}

    def evaluate(
        self,
        model: Any,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
        model_name: str = "model",
    ) -> EvaluationMetrics:
        """Evaluate a model and store results.

        Args:
            model: Trained model with predict and optionally predict_proba methods.
            X: Test features.
            y: True labels.
            model_name: Name for identifying this model.

        Returns:
            EvaluationMetrics for this model.
        """
        y_pred = model.predict(X)

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)

        metrics = compute_metrics(y, y_pred, y_proba, self.class_names)
        self.results[model_name] = metrics
        self._predictions[model_name] = (y, y_pred, y_proba)

        logger.info(f"Evaluated {model_name}: {metrics}")
        return metrics

    def plot_confusion_matrix(
        self,
        model_name: str,
        normalize: bool = True,
    ) -> plt.Figure:
        """Plot confusion matrix for a model."""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' has not been evaluated")

        metrics = self.results[model_name]
        save_path = None
        if self.figures_dir:
            save_path = self.figures_dir / f"confusion_matrix_{model_name}.png"

        return plot_confusion_matrix(
            metrics.confusion_matrix,
            self.class_names,
            normalize=normalize,
            title=f"Confusion Matrix - {model_name}",
            save_path=save_path,
        )

    def plot_roc_curves(self, model_name: str) -> plt.Figure | None:
        """Plot ROC curves for a model."""
        if model_name not in self._predictions:
            raise ValueError(f"Model '{model_name}' has not been evaluated")

        y_true, _, y_proba = self._predictions[model_name]
        if y_proba is None:
            logger.warning(f"Model '{model_name}' doesn't have probability predictions")
            return None

        save_path = None
        if self.figures_dir:
            save_path = self.figures_dir / f"roc_curves_{model_name}.png"

        return plot_roc_curves(
            y_true,
            y_proba,
            self.class_names,
            title=f"ROC Curves - {model_name}",
            save_path=save_path,
        )

    def plot_per_class_metrics(self, model_name: str) -> plt.Figure:
        """Plot per-class precision, recall, F1 for a model."""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' has not been evaluated")

        save_path = None
        if self.figures_dir:
            save_path = self.figures_dir / f"per_class_metrics_{model_name}.png"

        return plot_precision_recall_per_class(
            self.results[model_name],
            self.class_names,
            title=f"Per-Class Metrics - {model_name}",
            save_path=save_path,
        )

    def plot_comparison(self) -> plt.Figure:
        """Plot comparison of all evaluated models."""
        if not self.results:
            raise ValueError("No models have been evaluated")

        save_path = None
        if self.figures_dir:
            save_path = self.figures_dir / "model_comparison.png"

        return plot_model_comparison(
            self.results,
            save_path=save_path,
        )

    def plot_all(self, model_name: str | None = None) -> None:
        """Generate all plots for one or all models.

        Args:
            model_name: Specific model to plot, or None for all models.
        """
        if self.figures_dir:
            self.figures_dir.mkdir(parents=True, exist_ok=True)

        models_to_plot = [model_name] if model_name else list(self.results.keys())

        for name in models_to_plot:
            self.plot_confusion_matrix(name)
            self.plot_roc_curves(name)
            self.plot_per_class_metrics(name)

        if len(self.results) > 1:
            self.plot_comparison()

    def get_summary(self) -> str:
        """Get a text summary of all evaluation results."""
        lines = ["=" * 60, "Model Evaluation Summary", "=" * 60]

        for name, metrics in self.results.items():
            lines.append(f"\n{name}:")
            lines.append(str(metrics))

        if len(self.results) > 1:
            lines.append("\n" + "-" * 60)
            lines.append("Best Model by Metric:")

            metrics_to_compare = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
            for metric in metrics_to_compare:
                best_model = max(self.results.keys(), key=lambda x: getattr(self.results[x], metric))
                best_score = getattr(self.results[best_model], metric)
                lines.append(f"  {metric}: {best_model} ({best_score:.4f})")

        lines.append("=" * 60)
        return "\n".join(lines)
