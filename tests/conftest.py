"""Pytest fixtures and configuration for FitTrack ML tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Generate sample feature data matching HAR dataset structure."""
    np.random.seed(42)
    n_samples = 100
    n_features = 561

    # Generate random features in realistic range
    data = np.random.randn(n_samples, n_features) * 0.5

    # Create feature names matching HAR format
    feature_names = [f"feature_{i}" for i in range(n_features)]

    return pd.DataFrame(data, columns=feature_names)


@pytest.fixture
def sample_labels() -> pd.DataFrame:
    """Generate sample label data matching HAR dataset structure."""
    np.random.seed(42)
    n_samples = 100

    activity_ids = np.random.randint(1, 7, size=n_samples)
    activity_map = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
    }

    return pd.DataFrame({
        "activity_id": activity_ids,
        "activity": [activity_map[i] for i in activity_ids],
    })


@pytest.fixture
def sample_subject_ids() -> np.ndarray:
    """Generate sample subject IDs."""
    np.random.seed(42)
    return np.random.randint(1, 31, size=100)


@pytest.fixture
def mock_har_dataset(
    tmp_path: Path,
    sample_features: pd.DataFrame,
    sample_labels: pd.DataFrame,
    sample_subject_ids: np.ndarray,
) -> Path:
    """Create a mock HAR dataset directory structure for testing."""
    # Create directory structure
    dataset_dir = tmp_path / "UCI HAR Dataset"
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # Create feature names file
    feature_names = [f"feature_{i}" for i in range(561)]
    features_txt = "\n".join(f"{i+1} {name}" for i, name in enumerate(feature_names))
    (dataset_dir / "features.txt").write_text(features_txt)

    # Save train data
    sample_features.to_csv(
        train_dir / "X_train.txt",
        sep=" ",
        header=False,
        index=False,
    )
    sample_labels[["activity_id"]].to_csv(
        train_dir / "y_train.txt",
        sep=" ",
        header=False,
        index=False,
    )
    pd.Series(sample_subject_ids).to_csv(
        train_dir / "subject_train.txt",
        header=False,
        index=False,
    )

    # Save test data (use same data for simplicity)
    sample_features.to_csv(
        test_dir / "X_test.txt",
        sep=" ",
        header=False,
        index=False,
    )
    sample_labels[["activity_id"]].to_csv(
        test_dir / "y_test.txt",
        sep=" ",
        header=False,
        index=False,
    )
    pd.Series(sample_subject_ids).to_csv(
        test_dir / "subject_test.txt",
        header=False,
        index=False,
    )

    return dataset_dir
