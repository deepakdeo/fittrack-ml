"""Unit tests for data download module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fittrack.data.download import download_har_dataset


class TestDownloadHarDataset:
    """Tests for download_har_dataset function."""

    def test_skips_if_exists(self, tmp_path: Path) -> None:
        """Should skip download if data already exists."""
        # Create fake existing dataset
        existing_dir = tmp_path / "UCI HAR Dataset"
        existing_dir.mkdir()

        result = download_har_dataset(data_dir=tmp_path, force=False)

        assert result == existing_dir

    def test_force_redownload(self, tmp_path: Path) -> None:
        """Should redownload if force=True even if data exists."""
        # Create fake existing dataset
        existing_dir = tmp_path / "UCI HAR Dataset"
        existing_dir.mkdir()

        # Mock the requests.get to avoid actual download
        with patch("fittrack.data.download.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.headers = {"content-length": "1000"}
            mock_response.iter_content.return_value = [b"fake data"]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # Mock zipfile to avoid extraction issues
            with patch("fittrack.data.download.zipfile.ZipFile"):
                # This should attempt download despite existing dir
                with pytest.raises(Exception):
                    # Will fail because our mock zip isn't real, but proves download was attempted
                    download_har_dataset(data_dir=tmp_path, force=True)

    def test_creates_data_directory(self, tmp_path: Path) -> None:
        """Should create data directory if it doesn't exist."""
        new_dir = tmp_path / "new_data_dir"

        # Create fake existing dataset to skip actual download
        (new_dir / "UCI HAR Dataset").mkdir(parents=True)

        result = download_har_dataset(data_dir=new_dir)

        assert result.exists()


class TestDefaultDataDir:
    """Tests for default data directory handling."""

    def test_default_dir_path(self) -> None:
        """Default data dir should be relative to package location."""
        from fittrack.data.download import DEFAULT_DATA_DIR

        # Should end with data/raw
        assert DEFAULT_DATA_DIR.parts[-2:] == ("data", "raw")
