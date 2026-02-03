"""Data ingestion module for loading and validating the UCI HAR dataset.

This module provides utilities to load the train and test splits of the
UCI HAR dataset, perform basic validation, and return clean DataFrames.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series

logger = logging.getLogger(__name__)

# Activity labels mapping
ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "UCI HAR Dataset"


class HARFeatureSchema(pa.DataFrameModel):
    """Pandera schema for HAR feature data validation."""

    class Config:
        """Schema configuration."""

        strict = False  # Allow extra columns (561 features)
        coerce = True

    # We validate a subset of key features exist
    # The actual dataset has 561 features with names like tBodyAcc-mean()-X


class HARLabelSchema(pa.DataFrameModel):
    """Pandera schema for HAR label data validation."""

    activity_id: Series[int] = pa.Field(ge=1, le=6, description="Activity class ID")
    activity: Series[str] = pa.Field(isin=list(ACTIVITY_LABELS.values()))


@dataclass
class HARDataset:
    """Container for HAR dataset split.

    Attributes:
        X: Feature DataFrame with shape (n_samples, 561).
        y: Label DataFrame with activity_id and activity columns.
        subject_ids: Array of subject IDs for each sample.
    """

    X: pd.DataFrame
    y: pd.DataFrame
    subject_ids: np.ndarray

    @property
    def n_samples(self) -> int:
        """Return number of samples."""
        return len(self.X)

    @property
    def n_features(self) -> int:
        """Return number of features."""
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        """Return number of unique classes."""
        return self.y["activity_id"].nunique()


class HARDataLoader:
    """Load and validate the UCI HAR dataset.

    The UCI HAR dataset contains accelerometer and gyroscope readings from
    Samsung Galaxy S II smartphones. The data was collected from 30 subjects
    performing 6 different activities.

    Example:
        >>> loader = HARDataLoader("/path/to/UCI HAR Dataset")
        >>> train_data = loader.load_split("train")
        >>> print(f"Training samples: {train_data.n_samples}")
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        """Initialize the data loader.

        Args:
            data_dir: Path to the UCI HAR Dataset directory.
                     Defaults to data/raw/UCI HAR Dataset.

        Raises:
            FileNotFoundError: If the data directory doesn't exist.
        """
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}. "
                "Run `fittrack-download` first to download the dataset."
            )

        self._feature_names: list[str] | None = None

    @property
    def feature_names(self) -> list[str]:
        """Load and return feature names from features.txt.

        Note: The UCI HAR dataset has duplicate feature names, so we
        append indices to make them unique.
        """
        if self._feature_names is None:
            features_path = self.data_dir / "features.txt"
            with open(features_path) as f:
                # Format: "1 tBodyAcc-mean()-X"
                raw_names = [line.split()[1] for line in f.readlines()]

            # Make names unique by appending index for duplicates
            seen: dict[str, int] = {}
            unique_names: list[str] = []
            for name in raw_names:
                if name in seen:
                    seen[name] += 1
                    unique_names.append(f"{name}_{seen[name]}")
                else:
                    seen[name] = 0
                    unique_names.append(name)
            self._feature_names = unique_names
        return self._feature_names

    def load_split(
        self,
        split: Literal["train", "test"],
        validate: bool = True,
    ) -> HARDataset:
        """Load a dataset split (train or test).

        Args:
            split: Which split to load ("train" or "test").
            validate: Whether to validate data against schema.

        Returns:
            HARDataset containing features, labels, and subject IDs.

        Raises:
            ValueError: If split is not "train" or "test".
            FileNotFoundError: If data files are missing.
            pandera.errors.SchemaError: If validation fails.
        """
        if split not in ("train", "test"):
            raise ValueError(f"Split must be 'train' or 'test', got '{split}'")

        split_dir = self.data_dir / split
        logger.info(f"Loading {split} split from {split_dir}")

        # Load features
        X_path = split_dir / f"X_{split}.txt"
        X = pd.read_csv(X_path, sep=r"\s+", header=None, names=self.feature_names)
        logger.info(f"Loaded features: {X.shape}")

        # Load labels
        y_path = split_dir / f"y_{split}.txt"
        y = pd.read_csv(y_path, sep=r"\s+", header=None, names=["activity_id"])
        y["activity"] = y["activity_id"].map(ACTIVITY_LABELS)

        # Load subject IDs
        subject_path = split_dir / f"subject_{split}.txt"
        subject_ids = pd.read_csv(subject_path, header=None)[0].values

        if validate:
            self._validate_data(X, y)

        return HARDataset(X=X, y=y, subject_ids=subject_ids)

    def _validate_data(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Validate feature and label data.

        Args:
            X: Feature DataFrame to validate.
            y: Label DataFrame to validate.

        Raises:
            pandera.errors.SchemaError: If validation fails.
        """
        logger.info("Validating data against schema...")

        # Validate labels
        HARLabelSchema.validate(y)

        # Basic feature validation
        assert X.shape[1] == 561, f"Expected 561 features, got {X.shape[1]}"
        assert not X.isnull().any().any(), "Features contain null values"
        assert len(X) == len(y), "Feature and label counts don't match"

        logger.info("Data validation passed")

    def load_all(self, validate: bool = True) -> tuple[HARDataset, HARDataset]:
        """Load both train and test splits.

        Args:
            validate: Whether to validate data against schema.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        train = self.load_split("train", validate=validate)
        test = self.load_split("test", validate=validate)
        return train, test


def load_har_data(
    data_dir: Path | str | None = None,
    split: Literal["train", "test", "both"] = "both",
    validate: bool = True,
) -> HARDataset | tuple[HARDataset, HARDataset]:
    """Convenience function to load HAR data.

    Args:
        data_dir: Path to the UCI HAR Dataset directory.
        split: Which split(s) to load.
        validate: Whether to validate data.

    Returns:
        HARDataset if split is "train" or "test",
        tuple of (train, test) if split is "both".

    Example:
        >>> train, test = load_har_data()
        >>> print(f"Train: {train.n_samples}, Test: {test.n_samples}")
    """
    loader = HARDataLoader(data_dir)

    if split == "both":
        return loader.load_all(validate=validate)
    return loader.load_split(split, validate=validate)
