"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from fittrack.features.engineering import (
    FeatureConfig,
    FeatureExtractor,
    compute_energy,
    compute_entropy,
    compute_iqr,
    compute_sma,
    compute_zero_crossing_rate,
    create_windows,
    extract_frequency_domain_features,
    extract_har_features,
    extract_time_domain_features,
    get_feature_importance_groups,
)


class TestBasicFeatures:
    """Tests for basic feature computation functions."""

    def test_compute_sma(self) -> None:
        """Test Signal Magnitude Area computation."""
        # Simple case: all ones
        data = np.ones((10, 3))
        assert compute_sma(data) == pytest.approx(1.0)

        # Mixed signs
        data = np.array([[1, -1], [1, -1]])
        assert compute_sma(data) == pytest.approx(1.0)

        # 1D input
        data_1d = np.array([1, -1, 1, -1])
        assert compute_sma(data_1d) == pytest.approx(1.0)

    def test_compute_zero_crossing_rate(self) -> None:
        """Test zero-crossing rate computation."""
        # Alternating signal
        signal = np.array([1, -1, 1, -1, 1])
        zcr = compute_zero_crossing_rate(signal)
        assert zcr == pytest.approx(1.0)  # Every transition crosses zero

        # No crossings
        signal = np.array([1, 2, 3, 4, 5])
        zcr = compute_zero_crossing_rate(signal)
        assert zcr == pytest.approx(0.0)

        # Single value
        signal = np.array([1])
        zcr = compute_zero_crossing_rate(signal)
        assert zcr == pytest.approx(0.0)

    def test_compute_energy(self) -> None:
        """Test signal energy computation."""
        signal = np.array([1, 1, 1, 1])
        energy = compute_energy(signal)
        assert energy == pytest.approx(1.0)

        signal = np.array([2, 2])
        energy = compute_energy(signal)
        assert energy == pytest.approx(4.0)

    def test_compute_iqr(self) -> None:
        """Test interquartile range computation."""
        signal = np.arange(100)  # 0 to 99
        iqr = compute_iqr(signal)
        # Q75 = 74.25, Q25 = 24.75, IQR = 49.5
        assert iqr == pytest.approx(49.5, rel=0.01)

    def test_compute_entropy(self) -> None:
        """Test entropy computation."""
        # Uniform distribution should have higher entropy
        uniform = np.random.uniform(0, 1, 1000)
        # Concentrated distribution
        concentrated = np.array([0.5] * 900 + [0.1] * 100)

        entropy_uniform = compute_entropy(uniform)
        entropy_concentrated = compute_entropy(concentrated)

        assert entropy_uniform > entropy_concentrated


class TestTimeDomainFeatures:
    """Tests for time-domain feature extraction."""

    def test_extract_time_domain_features_shape(self) -> None:
        """Test that correct number of features are extracted."""
        window = np.random.randn(128, 3)
        features = extract_time_domain_features(window, ["X", "Y", "Z"])

        # Per-axis features: mean, std, min, max, range, median, zcr, energy, iqr, entropy = 10
        # 3 axes = 30 per-axis features
        # Cross-axis: sma, magnitude_mean, magnitude_std = 3
        # Total = 33
        assert len(features) == 33

    def test_extract_time_domain_features_1d(self) -> None:
        """Test with 1D input."""
        window = np.random.randn(128)
        features = extract_time_domain_features(window, ["X"])
        assert "mean_X" in features
        assert "std_X" in features

    def test_extract_time_domain_features_values(self) -> None:
        """Test that features have expected values."""
        # Known signal: constant
        window = np.ones((100, 3))
        features = extract_time_domain_features(window, ["X", "Y", "Z"])

        assert features["mean_X"] == pytest.approx(1.0)
        assert features["std_X"] == pytest.approx(0.0)
        assert features["min_X"] == pytest.approx(1.0)
        assert features["max_X"] == pytest.approx(1.0)
        assert features["zcr_X"] == pytest.approx(0.0)


class TestFrequencyDomainFeatures:
    """Tests for frequency-domain feature extraction."""

    def test_extract_frequency_features_shape(self) -> None:
        """Test that correct number of features are extracted."""
        window = np.random.randn(128, 3)
        features = extract_frequency_domain_features(
            window, sampling_rate=50.0, n_coeffs=5, axis_names=["X", "Y", "Z"]
        )

        # Per axis: dominant_freq, spectral_energy, n_coeffs, spectral_entropy, spectral_centroid
        # = 1 + 1 + 5 + 1 + 1 = 9 per axis
        # 3 axes = 27
        assert len(features) == 27

    def test_detect_dominant_frequency(self) -> None:
        """Test that dominant frequency is correctly detected."""
        # Create a 5Hz sine wave
        t = np.linspace(0, 1, 128)
        signal = np.sin(2 * np.pi * 5 * t).reshape(-1, 1)

        features = extract_frequency_domain_features(
            signal, sampling_rate=128.0, n_coeffs=5, axis_names=["X"]
        )

        # Dominant frequency should be close to 5Hz
        assert features["dominant_freq_X"] == pytest.approx(5.0, abs=1.0)


class TestWindowing:
    """Tests for windowing function."""

    def test_create_windows_basic(self) -> None:
        """Test basic windowing."""
        data = np.arange(256).reshape(-1, 1)
        windows = create_windows(data, window_size=64, overlap=0.5)

        # (256 - 64) / 32 + 1 = 7 windows
        assert windows.shape == (7, 64, 1)

    def test_create_windows_no_overlap(self) -> None:
        """Test windowing without overlap."""
        data = np.arange(256).reshape(-1, 1)
        windows = create_windows(data, window_size=64, overlap=0.0)

        # 256 / 64 = 4 windows
        assert windows.shape == (4, 64, 1)

    def test_create_windows_high_overlap(self) -> None:
        """Test windowing with high overlap."""
        data = np.arange(256).reshape(-1, 1)
        windows = create_windows(data, window_size=64, overlap=0.75)

        # (256 - 64) / 16 + 1 = 13 windows
        assert windows.shape == (13, 64, 1)

    def test_create_windows_1d(self) -> None:
        """Test windowing with 1D input."""
        data = np.arange(256)
        windows = create_windows(data, window_size=64, overlap=0.5)
        assert windows.shape == (7, 64, 1)

    def test_create_windows_too_short(self) -> None:
        """Test that error is raised for data shorter than window."""
        data = np.arange(32).reshape(-1, 1)
        with pytest.raises(ValueError, match="less than window size"):
            create_windows(data, window_size=64, overlap=0.5)

    def test_create_windows_content(self) -> None:
        """Test that window content is correct."""
        data = np.arange(128).reshape(-1, 1)
        windows = create_windows(data, window_size=64, overlap=0.5)

        # First window should be 0-63
        assert windows[0, 0, 0] == 0
        assert windows[0, -1, 0] == 63

        # Second window (50% overlap) should be 32-95
        assert windows[1, 0, 0] == 32
        assert windows[1, -1, 0] == 95


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class."""

    def test_default_config(self) -> None:
        """Test extractor with default config."""
        extractor = FeatureExtractor()
        assert extractor.config.include_time_domain is True
        assert extractor.config.include_frequency_domain is True

    def test_custom_config(self) -> None:
        """Test extractor with custom config."""
        config = FeatureConfig(
            include_time_domain=True,
            include_frequency_domain=False,
            window_size=64,
            window_overlap=0.25,
        )
        extractor = FeatureExtractor(config)
        assert extractor.config.window_size == 64
        assert extractor.config.include_frequency_domain is False

    def test_extract_window_features_time_only(self) -> None:
        """Test extracting only time-domain features."""
        config = FeatureConfig(include_time_domain=True, include_frequency_domain=False)
        extractor = FeatureExtractor(config)

        window = np.random.randn(128, 3)
        features = extractor.extract_window_features(window, ["X", "Y", "Z"])

        assert "mean_X" in features
        assert "dominant_freq_X" not in features

    def test_extract_window_features_freq_only(self) -> None:
        """Test extracting only frequency-domain features."""
        config = FeatureConfig(include_time_domain=False, include_frequency_domain=True)
        extractor = FeatureExtractor(config)

        window = np.random.randn(128, 3)
        features = extractor.extract_window_features(window, ["X", "Y", "Z"])

        assert "mean_X" not in features
        assert "dominant_freq_X" in features

    def test_extract_features_from_raw_data(self) -> None:
        """Test full feature extraction pipeline."""
        config = FeatureConfig(
            window_size=64,
            window_overlap=0.5,
            n_fft_coeffs=5,
        )
        extractor = FeatureExtractor(config)

        data = np.random.randn(256, 3)
        features_df = extractor.extract_features(data, ["X", "Y", "Z"])

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        assert "mean_X" in features_df.columns

    def test_extract_features_with_labels(self) -> None:
        """Test feature extraction with label assignment."""
        config = FeatureConfig(window_size=64, window_overlap=0.5)
        extractor = FeatureExtractor(config)

        data = np.random.randn(256, 3)
        labels = np.array([0] * 128 + [1] * 128)

        features_df = extractor.extract_features(data, labels=labels)

        assert "label" in features_df.columns
        assert features_df["label"].nunique() == 2


class TestHARFeatureExtraction:
    """Tests for UCI HAR specific feature extraction."""

    @pytest.fixture
    def sample_har_features(self) -> pd.DataFrame:
        """Create sample HAR-like feature DataFrame."""
        # Create feature names similar to UCI HAR
        feature_names = [
            "tBodyAcc-mean()-X",
            "tBodyAcc-mean()-Y",
            "tBodyAcc-mean()-Z",
            "tBodyAcc-std()-X",
            "tGravityAcc-mean()-X",
            "tBodyGyro-mean()-X",
            "fBodyAcc-mean()-X",
        ]
        data = np.random.randn(100, len(feature_names))
        return pd.DataFrame(data, columns=feature_names)

    def test_extract_all_features(self, sample_har_features: pd.DataFrame) -> None:
        """Test extracting all features."""
        result = extract_har_features(sample_har_features, feature_subset="all")
        assert len(result.columns) == len(sample_har_features.columns)

    def test_extract_time_features(self, sample_har_features: pd.DataFrame) -> None:
        """Test extracting time-domain features."""
        result = extract_har_features(sample_har_features, feature_subset="time")
        assert all(col.startswith("t") for col in result.columns)

    def test_extract_body_acc_features(self, sample_har_features: pd.DataFrame) -> None:
        """Test extracting body acceleration features."""
        result = extract_har_features(sample_har_features, feature_subset="body_acc")
        assert all("BodyAcc" in col for col in result.columns)

    def test_extract_gyro_features(self, sample_har_features: pd.DataFrame) -> None:
        """Test extracting gyroscope features."""
        result = extract_har_features(sample_har_features, feature_subset="gyro")
        assert all("Gyro" in col for col in result.columns)

    def test_invalid_subset(self, sample_har_features: pd.DataFrame) -> None:
        """Test that invalid subset raises error."""
        with pytest.raises(ValueError, match="Unknown feature_subset"):
            extract_har_features(sample_har_features, feature_subset="invalid")  # type: ignore


class TestFeatureImportanceGroups:
    """Tests for feature importance grouping."""

    def test_get_feature_importance_groups(self) -> None:
        """Test that feature groups are returned."""
        groups = get_feature_importance_groups()

        assert isinstance(groups, dict)
        assert "body_acceleration" in groups
        assert "gravity_acceleration" in groups
        assert "body_gyroscope" in groups
        assert "frequency_domain" in groups
