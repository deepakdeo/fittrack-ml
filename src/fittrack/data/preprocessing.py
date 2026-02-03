"""Data preprocessing module for HAR data.

This module provides utilities for preprocessing sensor data including
normalization, train/val/test splitting, and parallel processing with Dask.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Container for train/val/test split data.

    Attributes:
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.
        y_train: Training labels.
        y_val: Validation labels.
        y_test: Test labels.
        label_encoder: Fitted LabelEncoder for activity labels.
        scaler: Fitted StandardScaler for features.
    """

    X_train: NDArray[np.floating]
    X_val: NDArray[np.floating]
    X_test: NDArray[np.floating]
    y_train: NDArray[np.integer]
    y_val: NDArray[np.integer]
    y_test: NDArray[np.integer]
    label_encoder: LabelEncoder
    scaler: StandardScaler | None = None

    @property
    def n_classes(self) -> int:
        """Return number of classes."""
        return len(self.label_encoder.classes_)

    @property
    def n_features(self) -> int:
        """Return number of features."""
        return self.X_train.shape[1]

    @property
    def class_names(self) -> list[str]:
        """Return list of class names."""
        return list(self.label_encoder.classes_)


def normalize_features(
    X_train: pd.DataFrame | NDArray[np.floating],
    X_val: pd.DataFrame | NDArray[np.floating] | None = None,
    X_test: pd.DataFrame | NDArray[np.floating] | None = None,
    method: Literal["standard", "minmax"] = "standard",
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None, NDArray[np.floating] | None, StandardScaler]:
    """Normalize features using the specified method.

    Fits the scaler on training data only, then transforms all splits.

    Args:
        X_train: Training features.
        X_val: Validation features (optional).
        X_test: Test features (optional).
        method: Normalization method ("standard" for z-score, "minmax" for 0-1 scaling).

    Returns:
        Tuple of (normalized_train, normalized_val, normalized_test, fitted_scaler).

    Example:
        >>> X_train = np.random.randn(100, 10)
        >>> X_test = np.random.randn(50, 10)
        >>> X_train_norm, _, X_test_norm, scaler = normalize_features(X_train, X_test=X_test)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()  # type: ignore
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values

    # Fit on training data only
    X_train_normalized = scaler.fit_transform(X_train)
    logger.info(f"Fitted {method} scaler on training data with shape {X_train.shape}")

    X_val_normalized = None
    X_test_normalized = None

    if X_val is not None:
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        X_val_normalized = scaler.transform(X_val)

    if X_test is not None:
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        X_test_normalized = scaler.transform(X_test)

    return X_train_normalized, X_val_normalized, X_test_normalized, scaler  # type: ignore


def encode_labels(
    y: pd.Series | NDArray[np.str_],
) -> tuple[NDArray[np.integer], LabelEncoder]:
    """Encode string labels to integers.

    Args:
        y: Array or Series of string labels.

    Returns:
        Tuple of (encoded_labels, fitted_encoder).

    Example:
        >>> y = pd.Series(["WALKING", "SITTING", "WALKING"])
        >>> y_encoded, encoder = encode_labels(y)
        >>> y_encoded
        array([1, 0, 1])
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    logger.info(f"Encoded {len(encoder.classes_)} classes: {list(encoder.classes_)}")
    return y_encoded, encoder


def create_train_val_test_split(
    X: pd.DataFrame | NDArray[np.floating],
    y: pd.DataFrame | pd.Series | NDArray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
    normalize: bool = True,
) -> SplitData:
    """Create train/validation/test splits with optional normalization.

    Args:
        X: Feature matrix.
        y: Labels (DataFrame with 'activity' column, Series, or array).
        val_size: Fraction of data for validation.
        test_size: Fraction of data for test.
        random_state: Random seed for reproducibility.
        stratify: Whether to maintain class proportions in splits.
        normalize: Whether to apply standard scaling.

    Returns:
        SplitData containing all splits and fitted transformers.

    Example:
        >>> from fittrack.data.ingestion import load_har_data
        >>> train, test = load_har_data()
        >>> split = create_train_val_test_split(train.X, train.y)
        >>> print(f"Train: {len(split.X_train)}, Val: {len(split.X_val)}")
    """
    # Extract labels if DataFrame
    if isinstance(y, pd.DataFrame):
        if "activity" in y.columns:
            y_labels = y["activity"].values
        elif "activity_id" in y.columns:
            y_labels = y["activity_id"].values
        else:
            y_labels = y.iloc[:, 0].values
    elif isinstance(y, pd.Series):
        y_labels = y.values
    else:
        y_labels = y

    # Convert X to numpy
    X_array = X.values if isinstance(X, pd.DataFrame) else X

    # Encode labels
    y_encoded, label_encoder = encode_labels(y_labels)

    # First split: train+val vs test
    stratify_split = y_encoded if stratify else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_array,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_split,
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)  # Adjust for remaining data
    stratify_split2 = y_trainval if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_split2,
    )

    logger.info(
        f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # Normalize if requested
    scaler = None
    if normalize:
        X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
        logger.info("Applied StandardScaler normalization")

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_encoder=label_encoder,
        scaler=scaler,
    )


def create_subject_split(
    X: pd.DataFrame | NDArray[np.floating],
    y: pd.DataFrame | pd.Series | NDArray,
    subject_ids: NDArray[np.integer],
    test_subjects: list[int] | None = None,
    val_subjects: list[int] | None = None,
    random_state: int = 42,
    normalize: bool = True,
) -> SplitData:
    """Create train/val/test splits by subject (no data leakage across subjects).

    This is the recommended approach for HAR data to ensure the model
    generalizes to unseen subjects.

    Args:
        X: Feature matrix.
        y: Labels.
        subject_ids: Array of subject IDs for each sample.
        test_subjects: Specific subjects for test set. If None, randomly select ~20%.
        val_subjects: Specific subjects for validation set. If None, randomly select ~20%.
        random_state: Random seed for reproducibility.
        normalize: Whether to apply standard scaling.

    Returns:
        SplitData containing all splits and fitted transformers.

    Example:
        >>> split = create_subject_split(X, y, subject_ids)
        >>> print(f"Train subjects: {len(np.unique(train_subjects))}")
    """
    np.random.seed(random_state)
    unique_subjects = np.unique(subject_ids)

    if test_subjects is None:
        n_test = max(1, int(len(unique_subjects) * 0.2))
        test_subjects = list(np.random.choice(unique_subjects, n_test, replace=False))

    remaining_subjects = [s for s in unique_subjects if s not in test_subjects]

    if val_subjects is None:
        n_val = max(1, int(len(remaining_subjects) * 0.2))
        val_subjects = list(np.random.choice(remaining_subjects, n_val, replace=False))

    train_subjects = [s for s in remaining_subjects if s not in val_subjects]

    logger.info(
        f"Subject split - Train: {len(train_subjects)}, "
        f"Val: {len(val_subjects)}, Test: {len(test_subjects)}"
    )

    # Create masks
    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    test_mask = np.isin(subject_ids, test_subjects)

    # Extract labels
    if isinstance(y, pd.DataFrame):
        y_labels = y["activity"].values if "activity" in y.columns else y.iloc[:, 0].values
    elif isinstance(y, pd.Series):
        y_labels = y.values
    else:
        y_labels = y

    X_array = X.values if isinstance(X, pd.DataFrame) else X

    # Encode labels
    y_encoded, label_encoder = encode_labels(y_labels)

    # Split data
    X_train = X_array[train_mask]
    X_val = X_array[val_mask]
    X_test = X_array[test_mask]
    y_train = y_encoded[train_mask]
    y_val = y_encoded[val_mask]
    y_test = y_encoded[test_mask]

    # Normalize
    scaler = None
    if normalize:
        X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_encoder=label_encoder,
        scaler=scaler,
    )


def get_class_weights(
    y: NDArray[np.integer],
    method: Literal["balanced", "sqrt"] = "balanced",
) -> dict[int, float]:
    """Compute class weights for imbalanced data.

    Args:
        y: Encoded labels.
        method: Weight computation method.
            - "balanced": Inverse frequency weighting.
            - "sqrt": Square root of inverse frequency (less aggressive).

    Returns:
        Dictionary mapping class index to weight.

    Example:
        >>> y = np.array([0, 0, 0, 1, 2, 2])
        >>> weights = get_class_weights(y)
        >>> weights[0]  # Most common class gets lowest weight
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)

    if method == "balanced":
        weights = n_samples / (n_classes * counts)
    elif method == "sqrt":
        weights = np.sqrt(n_samples / (n_classes * counts))
    else:
        raise ValueError(f"Unknown method: {method}")

    weight_dict = {int(c): float(w) for c, w in zip(classes, weights, strict=False)}
    logger.info(f"Computed class weights: {weight_dict}")
    return weight_dict


def get_sample_weights(
    y: NDArray[np.integer],
    class_weights: dict[int, float] | None = None,
) -> NDArray[np.floating]:
    """Get per-sample weights from class weights.

    Args:
        y: Encoded labels.
        class_weights: Pre-computed class weights. If None, computes balanced weights.

    Returns:
        Array of sample weights.
    """
    if class_weights is None:
        class_weights = get_class_weights(y)

    return np.array([class_weights[int(label)] for label in y])


def remove_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove highly correlated features.

    Args:
        X: Feature DataFrame.
        threshold: Correlation threshold above which to remove features.

    Returns:
        Tuple of (filtered DataFrame, list of removed feature names).

    Example:
        >>> X_filtered, removed = remove_correlated_features(X, threshold=0.95)
        >>> print(f"Removed {len(removed)} features")
    """
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    logger.info(f"Removing {len(to_drop)} correlated features (threshold={threshold})")

    return X.drop(columns=to_drop), to_drop


def select_top_features(
    X: pd.DataFrame,
    y: NDArray[np.integer],
    n_features: int = 100,
    method: Literal["mutual_info", "f_classif"] = "mutual_info",
) -> tuple[pd.DataFrame, list[str]]:
    """Select top features using feature selection.

    Args:
        X: Feature DataFrame.
        y: Encoded labels.
        n_features: Number of features to select.
        method: Selection method ("mutual_info" or "f_classif").

    Returns:
        Tuple of (selected features DataFrame, selected feature names).
    """
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

    if method == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=n_features)
    elif method == "f_classif":
        selector = SelectKBest(f_classif, k=n_features)
    else:
        raise ValueError(f"Unknown method: {method}")

    selector.fit(X, y)
    mask = selector.get_support()
    selected_features = X.columns[mask].tolist()

    logger.info(f"Selected {len(selected_features)} features using {method}")
    return X[selected_features], selected_features


class DataPreprocessor:
    """Preprocessing pipeline for HAR data.

    This class provides a unified interface for all preprocessing steps
    including normalization, encoding, and feature selection.

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> split_data = preprocessor.fit_transform(X, y)
        >>> X_new_normalized = preprocessor.transform(X_new)
    """

    def __init__(
        self,
        normalize: bool = True,
        remove_correlated: bool = False,
        correlation_threshold: float = 0.95,
        n_select_features: int | None = None,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            normalize: Whether to apply standard scaling.
            remove_correlated: Whether to remove highly correlated features.
            correlation_threshold: Threshold for correlation removal.
            n_select_features: Number of features to select (None = all).
        """
        self.normalize = normalize
        self.remove_correlated = remove_correlated
        self.correlation_threshold = correlation_threshold
        self.n_select_features = n_select_features

        self.scaler: StandardScaler | None = None
        self.label_encoder: LabelEncoder | None = None
        self.selected_features: list[str] | None = None
        self._is_fitted = False

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> SplitData:
        """Fit the preprocessor and transform data.

        Args:
            X: Feature DataFrame.
            y: Labels.
            val_size: Validation set fraction.
            test_size: Test set fraction.
            random_state: Random seed.

        Returns:
            SplitData with preprocessed splits.
        """
        # Feature selection/removal
        X_processed = X.copy()

        if self.remove_correlated:
            X_processed, _ = remove_correlated_features(
                X_processed, self.correlation_threshold
            )

        if self.n_select_features is not None:
            y_encoded, _ = encode_labels(
                y["activity"] if isinstance(y, pd.DataFrame) else y
            )
            X_processed, self.selected_features = select_top_features(
                X_processed, y_encoded, self.n_select_features
            )
        else:
            self.selected_features = X_processed.columns.tolist()

        # Create splits
        split_data = create_train_val_test_split(
            X_processed,
            y,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            normalize=self.normalize,
        )

        self.scaler = split_data.scaler
        self.label_encoder = split_data.label_encoder
        self._is_fitted = True

        return split_data

    def transform(self, X: pd.DataFrame) -> NDArray[np.floating]:
        """Transform new data using fitted preprocessor.

        Args:
            X: Feature DataFrame.

        Returns:
            Preprocessed feature array.

        Raises:
            ValueError: If preprocessor hasn't been fitted.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Select features
        if self.selected_features is not None:
            X_processed = X[self.selected_features].values
        else:
            X_processed = X.values

        # Normalize
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)

        return X_processed

    def inverse_transform_labels(self, y: NDArray[np.integer]) -> NDArray[np.str_]:
        """Convert encoded labels back to original strings.

        Args:
            y: Encoded labels.

        Returns:
            Original string labels.
        """
        if self.label_encoder is None:
            raise ValueError("Preprocessor must be fitted first")
        return self.label_encoder.inverse_transform(y)
