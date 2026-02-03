"""Unit tests for data ingestion module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fittrack.data.ingestion import (
    ACTIVITY_LABELS,
    HARDataLoader,
    HARDataset,
    load_har_data,
)


class TestActivityLabels:
    """Tests for activity label mapping."""

    def test_activity_labels_count(self) -> None:
        """Should have exactly 6 activity labels."""
        assert len(ACTIVITY_LABELS) == 6

    def test_activity_labels_ids(self) -> None:
        """Activity IDs should be 1-6."""
        assert set(ACTIVITY_LABELS.keys()) == {1, 2, 3, 4, 5, 6}

    def test_activity_labels_names(self) -> None:
        """Should contain expected activity names."""
        expected = {
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING",
        }
        assert set(ACTIVITY_LABELS.values()) == expected


class TestHARDataset:
    """Tests for HARDataset dataclass."""

    def test_dataset_properties(
        self,
        sample_features: pd.DataFrame,
        sample_labels: pd.DataFrame,
        sample_subject_ids: np.ndarray,
    ) -> None:
        """Dataset should correctly report its properties."""
        dataset = HARDataset(
            X=sample_features,
            y=sample_labels,
            subject_ids=sample_subject_ids,
        )

        assert dataset.n_samples == 100
        assert dataset.n_features == 561
        assert dataset.n_classes == 6


class TestHARDataLoader:
    """Tests for HARDataLoader class."""

    def test_init_with_missing_dir(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            HARDataLoader(tmp_path / "nonexistent")

    def test_init_with_valid_dir(self, mock_har_dataset: Path) -> None:
        """Should initialize successfully with valid directory."""
        loader = HARDataLoader(mock_har_dataset)
        assert loader.data_dir == mock_har_dataset

    def test_feature_names(self, mock_har_dataset: Path) -> None:
        """Should load feature names from features.txt."""
        loader = HARDataLoader(mock_har_dataset)
        assert len(loader.feature_names) == 561
        assert loader.feature_names[0] == "feature_0"

    def test_load_train_split(self, mock_har_dataset: Path) -> None:
        """Should load train split correctly."""
        loader = HARDataLoader(mock_har_dataset)
        dataset = loader.load_split("train")

        assert isinstance(dataset, HARDataset)
        assert dataset.n_samples == 100
        assert dataset.n_features == 561
        assert len(dataset.subject_ids) == 100

    def test_load_test_split(self, mock_har_dataset: Path) -> None:
        """Should load test split correctly."""
        loader = HARDataLoader(mock_har_dataset)
        dataset = loader.load_split("test")

        assert isinstance(dataset, HARDataset)
        assert dataset.n_samples == 100

    def test_load_invalid_split(self, mock_har_dataset: Path) -> None:
        """Should raise ValueError for invalid split."""
        loader = HARDataLoader(mock_har_dataset)
        with pytest.raises(ValueError, match="Split must be 'train' or 'test'"):
            loader.load_split("invalid")  # type: ignore

    def test_load_all(self, mock_har_dataset: Path) -> None:
        """Should load both train and test splits."""
        loader = HARDataLoader(mock_har_dataset)
        train, test = loader.load_all()

        assert isinstance(train, HARDataset)
        assert isinstance(test, HARDataset)

    def test_labels_have_activity_names(self, mock_har_dataset: Path) -> None:
        """Labels should include both activity_id and activity name."""
        loader = HARDataLoader(mock_har_dataset)
        dataset = loader.load_split("train")

        assert "activity_id" in dataset.y.columns
        assert "activity" in dataset.y.columns
        assert all(activity in ACTIVITY_LABELS.values() for activity in dataset.y["activity"])


class TestLoadHarData:
    """Tests for load_har_data convenience function."""

    def test_load_both_splits(self, mock_har_dataset: Path) -> None:
        """Should return tuple when split='both'."""
        train, test = load_har_data(mock_har_dataset, split="both")

        assert isinstance(train, HARDataset)
        assert isinstance(test, HARDataset)

    def test_load_single_split(self, mock_har_dataset: Path) -> None:
        """Should return single dataset when split is 'train' or 'test'."""
        train = load_har_data(mock_har_dataset, split="train")
        assert isinstance(train, HARDataset)

        test = load_har_data(mock_har_dataset, split="test")
        assert isinstance(test, HARDataset)

    def test_skip_validation(self, mock_har_dataset: Path) -> None:
        """Should work with validation disabled."""
        dataset = load_har_data(mock_har_dataset, split="train", validate=False)
        assert dataset.n_samples == 100


class TestDataValidation:
    """Tests for data validation functionality."""

    def test_validates_label_range(
        self,
        mock_har_dataset: Path,
        sample_features: pd.DataFrame,
    ) -> None:
        """Should validate activity_id is in range 1-6."""
        loader = HARDataLoader(mock_har_dataset)
        # The mock data is valid, so this should pass
        dataset = loader.load_split("train", validate=True)
        assert all(dataset.y["activity_id"].between(1, 6))

    def test_feature_count_validation(self, mock_har_dataset: Path) -> None:
        """Should verify 561 features are present."""
        loader = HARDataLoader(mock_har_dataset)
        dataset = loader.load_split("train", validate=True)
        assert dataset.n_features == 561
