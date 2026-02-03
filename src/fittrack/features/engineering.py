"""Feature engineering module for time-series sensor data.

This module provides utilities for extracting features from accelerometer and
gyroscope sensor data, including time-domain statistics, frequency-domain
features, and windowing functions for raw time-series data.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction.

    Attributes:
        include_time_domain: Whether to extract time-domain features.
        include_frequency_domain: Whether to extract frequency-domain (FFT) features.
        window_size: Size of the sliding window in samples.
        window_overlap: Fraction of overlap between consecutive windows (0-1).
        n_fft_coeffs: Number of FFT coefficients to include per axis.
    """

    include_time_domain: bool = True
    include_frequency_domain: bool = True
    window_size: int = 128
    window_overlap: float = 0.5
    n_fft_coeffs: int = 10


def compute_sma(data: NDArray[np.floating]) -> float:
    """Compute Signal Magnitude Area (SMA).

    SMA is the sum of absolute values across all axes, normalized by the number
    of samples. It's a measure of overall signal energy.

    Args:
        data: 2D array of shape (n_samples, n_axes) or 1D array.

    Returns:
        Signal Magnitude Area value.

    Example:
        >>> data = np.array([[1, 2, 3], [4, 5, 6]])
        >>> compute_sma(data)
        3.5
    """
    return float(np.mean(np.abs(data)))


def compute_zero_crossing_rate(signal: NDArray[np.floating]) -> float:
    """Compute zero-crossing rate of a 1D signal.

    Zero-crossing rate is the rate at which the signal changes sign. It's useful
    for distinguishing periodic from aperiodic signals.

    Args:
        signal: 1D array of signal values.

    Returns:
        Zero-crossing rate (crossings per sample).

    Example:
        >>> signal = np.array([1, -1, 1, -1, 1])
        >>> compute_zero_crossing_rate(signal)
        0.8
    """
    if len(signal) < 2:
        return 0.0
    crossings = np.sum(np.diff(np.sign(signal)) != 0)
    return float(crossings / (len(signal) - 1))


def compute_energy(signal: NDArray[np.floating]) -> float:
    """Compute signal energy (sum of squared values normalized by length).

    Args:
        signal: 1D or 2D array of signal values.

    Returns:
        Signal energy value.
    """
    return float(np.sum(signal**2) / signal.size)


def compute_iqr(signal: NDArray[np.floating]) -> float:
    """Compute interquartile range of a signal.

    Args:
        signal: 1D array of signal values.

    Returns:
        Interquartile range (Q3 - Q1).
    """
    q75, q25 = np.percentile(signal, [75, 25])
    return float(q75 - q25)


def compute_entropy(signal: NDArray[np.floating], bins: int = 10) -> float:
    """Compute signal entropy using histogram-based probability estimation.

    Args:
        signal: 1D array of signal values.
        bins: Number of histogram bins for probability estimation.

    Returns:
        Entropy value in nats.
    """
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zeros for log
    if len(hist) == 0:
        return 0.0
    # Normalize to get probabilities
    probs = hist / hist.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def extract_time_domain_features(
    window: NDArray[np.floating],
    axis_names: list[str] | None = None,
) -> dict[str, float]:
    """Extract time-domain features from a signal window.

    Extracts statistical features commonly used in activity recognition:
    - Mean, standard deviation, min, max per axis
    - Signal Magnitude Area (SMA)
    - Zero-crossing rate per axis
    - Energy, IQR, entropy per axis

    Args:
        window: 2D array of shape (n_samples, n_axes) or 1D array for single axis.
        axis_names: Names for each axis. Defaults to X, Y, Z or numbered.

    Returns:
        Dictionary mapping feature names to values.

    Example:
        >>> window = np.random.randn(128, 3)
        >>> features = extract_time_domain_features(window, ["X", "Y", "Z"])
        >>> "mean_X" in features
        True
    """
    # Ensure 2D
    if window.ndim == 1:
        window = window.reshape(-1, 1)

    n_axes = window.shape[1]
    if axis_names is None:
        axis_names = ["X", "Y", "Z"] if n_axes == 3 else [str(i) for i in range(n_axes)]

    features: dict[str, float] = {}

    # Per-axis features
    for i, axis in enumerate(axis_names):
        signal = window[:, i]
        features[f"mean_{axis}"] = float(np.mean(signal))
        features[f"std_{axis}"] = float(np.std(signal))
        features[f"min_{axis}"] = float(np.min(signal))
        features[f"max_{axis}"] = float(np.max(signal))
        features[f"range_{axis}"] = float(np.ptp(signal))
        features[f"median_{axis}"] = float(np.median(signal))
        features[f"zcr_{axis}"] = compute_zero_crossing_rate(signal)
        features[f"energy_{axis}"] = compute_energy(signal)
        features[f"iqr_{axis}"] = compute_iqr(signal)
        features[f"entropy_{axis}"] = compute_entropy(signal)

    # Cross-axis features
    features["sma"] = compute_sma(window)
    features["magnitude_mean"] = float(np.mean(np.linalg.norm(window, axis=1)))
    features["magnitude_std"] = float(np.std(np.linalg.norm(window, axis=1)))

    return features


def extract_frequency_domain_features(
    window: NDArray[np.floating],
    sampling_rate: float = 50.0,
    n_coeffs: int = 10,
    axis_names: list[str] | None = None,
) -> dict[str, float]:
    """Extract frequency-domain features using FFT.

    Computes FFT and extracts:
    - Dominant frequency per axis
    - Spectral energy per axis
    - First N FFT magnitude coefficients per axis
    - Spectral entropy per axis

    Args:
        window: 2D array of shape (n_samples, n_axes) or 1D array.
        sampling_rate: Sampling rate in Hz.
        n_coeffs: Number of FFT coefficients to include per axis.
        axis_names: Names for each axis.

    Returns:
        Dictionary mapping feature names to values.

    Example:
        >>> window = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 128)).reshape(-1, 1)
        >>> features = extract_frequency_domain_features(window, n_coeffs=5)
        >>> "fft_coeff_0_0" in features
        True
    """
    # Ensure 2D
    if window.ndim == 1:
        window = window.reshape(-1, 1)

    n_samples, n_axes = window.shape
    if axis_names is None:
        axis_names = ["X", "Y", "Z"] if n_axes == 3 else [str(i) for i in range(n_axes)]

    features: dict[str, float] = {}

    # Frequency bins
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)

    for i, axis in enumerate(axis_names):
        signal = window[:, i]

        # Compute FFT
        fft_vals = np.fft.rfft(signal)
        fft_magnitude = np.abs(fft_vals)

        # Dominant frequency
        dominant_idx = np.argmax(fft_magnitude[1:]) + 1  # Skip DC
        features[f"dominant_freq_{axis}"] = float(freqs[dominant_idx])

        # Spectral energy
        features[f"spectral_energy_{axis}"] = float(np.sum(fft_magnitude**2))

        # FFT coefficients (magnitude)
        for j in range(min(n_coeffs, len(fft_magnitude))):
            features[f"fft_coeff_{j}_{axis}"] = float(fft_magnitude[j])

        # Spectral entropy
        power_spectrum = fft_magnitude**2
        power_spectrum = power_spectrum / (power_spectrum.sum() + 1e-10)
        spectral_entropy = -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        features[f"spectral_entropy_{axis}"] = float(spectral_entropy)

        # Spectral centroid (weighted mean frequency)
        features[f"spectral_centroid_{axis}"] = float(
            np.sum(freqs * fft_magnitude) / (np.sum(fft_magnitude) + 1e-10)
        )

    return features


def create_windows(
    data: NDArray[np.floating],
    window_size: int = 128,
    overlap: float = 0.5,
) -> NDArray[np.floating]:
    """Create overlapping windows from time-series data.

    Args:
        data: 2D array of shape (n_samples, n_features) or 1D array.
        window_size: Number of samples per window.
        overlap: Fraction of overlap between consecutive windows (0-1).

    Returns:
        3D array of shape (n_windows, window_size, n_features).

    Raises:
        ValueError: If data is too short for the window size.

    Example:
        >>> data = np.arange(256).reshape(-1, 1)
        >>> windows = create_windows(data, window_size=64, overlap=0.5)
        >>> windows.shape
        (7, 64, 1)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_features = data.shape

    if n_samples < window_size:
        raise ValueError(f"Data length ({n_samples}) is less than window size ({window_size})")

    step_size = int(window_size * (1 - overlap))
    if step_size < 1:
        step_size = 1

    n_windows = (n_samples - window_size) // step_size + 1

    windows = np.zeros((n_windows, window_size, n_features), dtype=data.dtype)

    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[start:end]

    return windows


class FeatureExtractor:
    """Extract features from raw sensor time-series data.

    This class provides a unified interface for extracting time-domain and
    frequency-domain features from accelerometer and gyroscope data.

    Example:
        >>> extractor = FeatureExtractor(FeatureConfig())
        >>> raw_data = np.random.randn(1000, 6)  # 6-axis sensor
        >>> features_df = extractor.extract_features(
        ...     raw_data,
        ...     axis_names=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        ... )
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize the feature extractor.

        Args:
            config: Feature extraction configuration. Uses defaults if None.
        """
        self.config = config or FeatureConfig()

    def extract_window_features(
        self,
        window: NDArray[np.floating],
        axis_names: list[str] | None = None,
        sampling_rate: float = 50.0,
    ) -> dict[str, float]:
        """Extract all features from a single window.

        Args:
            window: 2D array of shape (n_samples, n_axes).
            axis_names: Names for each axis.
            sampling_rate: Sampling rate in Hz (for frequency features).

        Returns:
            Dictionary of all extracted features.
        """
        features: dict[str, float] = {}

        if self.config.include_time_domain:
            time_features = extract_time_domain_features(window, axis_names)
            features.update(time_features)

        if self.config.include_frequency_domain:
            freq_features = extract_frequency_domain_features(
                window,
                sampling_rate=sampling_rate,
                n_coeffs=self.config.n_fft_coeffs,
                axis_names=axis_names,
            )
            features.update(freq_features)

        return features

    def extract_features(
        self,
        data: NDArray[np.floating],
        axis_names: list[str] | None = None,
        sampling_rate: float = 50.0,
        labels: NDArray[np.integer] | None = None,
    ) -> pd.DataFrame:
        """Extract features from raw time-series data using sliding windows.

        Args:
            data: 2D array of shape (n_samples, n_axes) containing raw sensor data.
            axis_names: Names for each axis (e.g., ["acc_x", "acc_y", "acc_z"]).
            sampling_rate: Sampling rate of the data in Hz.
            labels: Optional per-sample labels. Will assign majority label to each window.

        Returns:
            DataFrame where each row contains features from one window.
        """
        logger.info(
            f"Extracting features from data shape {data.shape} "
            f"with window_size={self.config.window_size}, overlap={self.config.window_overlap}"
        )

        # Create windows
        windows = create_windows(
            data,
            window_size=self.config.window_size,
            overlap=self.config.window_overlap,
        )
        logger.info(f"Created {len(windows)} windows")

        # Extract features from each window
        all_features = []
        for window in windows:
            features = self.extract_window_features(window, axis_names, sampling_rate)
            all_features.append(features)

        features_df = pd.DataFrame(all_features)

        # Assign labels if provided
        if labels is not None:
            step_size = int(self.config.window_size * (1 - self.config.window_overlap))
            window_labels = []
            for i in range(len(windows)):
                start = i * step_size
                end = start + self.config.window_size
                # Majority vote for window label
                window_label_values, counts = np.unique(labels[start:end], return_counts=True)
                window_labels.append(window_label_values[np.argmax(counts)])
            features_df["label"] = window_labels

        logger.info(f"Extracted {len(features_df.columns)} features per window")
        return features_df


def extract_har_features(
    X: pd.DataFrame,
    feature_subset: Literal["all", "time", "body_acc", "gyro"] = "all",
) -> pd.DataFrame:
    """Extract a subset of features from the UCI HAR pre-computed features.

    This function filters the 561 pre-computed features in the UCI HAR dataset
    to return a specific subset for analysis or modeling.

    Args:
        X: DataFrame with UCI HAR features (561 columns).
        feature_subset: Which features to extract:
            - "all": All 561 features
            - "time": Only time-domain features (prefix "t")
            - "body_acc": Only body acceleration features
            - "gyro": Only gyroscope features

    Returns:
        DataFrame with filtered features.

    Example:
        >>> from fittrack.data.ingestion import load_har_data
        >>> train, test = load_har_data()
        >>> time_features = extract_har_features(train.X, feature_subset="time")
    """
    if feature_subset == "all":
        return X

    cols = X.columns.tolist()

    if feature_subset == "time":
        selected = [c for c in cols if c.startswith("t")]
    elif feature_subset == "body_acc":
        selected = [c for c in cols if "BodyAcc" in c]
    elif feature_subset == "gyro":
        selected = [c for c in cols if "Gyro" in c]
    else:
        raise ValueError(f"Unknown feature_subset: {feature_subset}")

    logger.info(f"Selected {len(selected)} features for subset '{feature_subset}'")
    return X[selected]


def get_feature_importance_groups() -> dict[str, list[str]]:
    """Get feature groups commonly used for importance analysis.

    Returns:
        Dictionary mapping group names to feature name patterns.
    """
    return {
        "body_acceleration": ["tBodyAcc", "fBodyAcc"],
        "gravity_acceleration": ["tGravityAcc"],
        "body_gyroscope": ["tBodyGyro", "fBodyGyro"],
        "jerk_signals": ["Jerk"],
        "magnitude": ["Mag"],
        "mean_features": ["-mean()"],
        "std_features": ["-std()"],
        "frequency_domain": ["fBody"],
    }
