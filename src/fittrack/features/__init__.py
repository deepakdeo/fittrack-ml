"""Feature engineering modules."""

from fittrack.features.engineering import (
    FeatureConfig,
    FeatureExtractor,
    compute_sma,
    compute_zero_crossing_rate,
    create_windows,
    extract_frequency_domain_features,
    extract_har_features,
    extract_time_domain_features,
)

__all__ = [
    "FeatureConfig",
    "FeatureExtractor",
    "compute_sma",
    "compute_zero_crossing_rate",
    "create_windows",
    "extract_frequency_domain_features",
    "extract_har_features",
    "extract_time_domain_features",
]
