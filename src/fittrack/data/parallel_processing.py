"""Parallel data processing with Dask.

This module demonstrates how to scale data processing pipelines
using Dask for larger-than-memory datasets.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


def create_local_cluster(
    n_workers: int = 4,
    threads_per_worker: int = 2,
    memory_limit: str = "2GB",
) -> Client:
    """Create a local Dask cluster.

    Args:
        n_workers: Number of worker processes.
        threads_per_worker: Threads per worker.
        memory_limit: Memory limit per worker.

    Returns:
        Dask distributed client.

    Example:
        >>> client = create_local_cluster(n_workers=4)
        >>> print(client.dashboard_link)
    """
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard: {client.dashboard_link}")
    return client


def load_csv_parallel(
    file_pattern: str | Path,
    **kwargs: Any,
) -> dd.DataFrame:
    """Load CSV files in parallel using Dask.

    Args:
        file_pattern: Glob pattern for CSV files.
        **kwargs: Additional arguments for dd.read_csv.

    Returns:
        Dask DataFrame.

    Example:
        >>> ddf = load_csv_parallel("data/raw/*.csv")
        >>> print(f"Partitions: {ddf.npartitions}")
    """
    ddf = dd.read_csv(str(file_pattern), **kwargs)
    logger.info(f"Loaded {ddf.npartitions} partitions")
    return ddf


def process_sensor_data_parallel(
    ddf: dd.DataFrame,
    feature_columns: list[str],
    window_size: int = 128,
    overlap: float = 0.5,
) -> dd.DataFrame:
    """Process sensor data with sliding windows in parallel.

    Args:
        ddf: Dask DataFrame with sensor readings.
        feature_columns: Columns containing sensor data.
        window_size: Samples per window.
        overlap: Window overlap fraction.

    Returns:
        Dask DataFrame with extracted features.
    """
    def extract_window_features(partition: pd.DataFrame) -> pd.DataFrame:
        """Extract features from a single partition."""
        features_list = []
        step = int(window_size * (1 - overlap))

        for start in range(0, len(partition) - window_size + 1, step):
            window = partition.iloc[start:start + window_size]
            window_features = {}

            for col in feature_columns:
                data = window[col].values
                window_features[f"{col}_mean"] = np.mean(data)
                window_features[f"{col}_std"] = np.std(data)
                window_features[f"{col}_min"] = np.min(data)
                window_features[f"{col}_max"] = np.max(data)

            features_list.append(window_features)

        return pd.DataFrame(features_list)

    # Apply to each partition
    meta = {f"{col}_{stat}": float for col in feature_columns for stat in ["mean", "std", "min", "max"]}
    result = ddf.map_partitions(extract_window_features, meta=meta)

    return result


def parallel_feature_engineering(
    df: pd.DataFrame,
    feature_func: Callable[[pd.DataFrame], pd.DataFrame],
    n_partitions: int = 4,
) -> pd.DataFrame:
    """Apply feature engineering in parallel.

    Args:
        df: Input DataFrame.
        feature_func: Function to apply to each partition.
        n_partitions: Number of partitions for parallel processing.

    Returns:
        DataFrame with extracted features.

    Example:
        >>> def my_features(df):
        ...     return df.assign(feature1=df['x'] ** 2)
        >>> result = parallel_feature_engineering(df, my_features)
    """
    # Convert to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=n_partitions)

    # Apply function in parallel
    result_ddf = ddf.map_partitions(feature_func)

    # Compute and return
    return result_ddf.compute()


def parallel_model_inference(
    model: Any,
    X: np.ndarray,
    batch_size: int = 1000,
    _n_partitions: int = 4,
) -> np.ndarray:
    """Run model inference in parallel using Dask.

    Args:
        model: Trained model with predict method.
        X: Input features.
        batch_size: Samples per batch.
        n_partitions: Number of parallel partitions.

    Returns:
        Array of predictions.

    Example:
        >>> predictions = parallel_model_inference(model, X_test)
    """
    import dask.array as da

    # Convert to Dask array
    X_dask = da.from_array(X, chunks=(batch_size, -1))

    # Define prediction function
    def predict_batch(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch)

    # Apply in parallel
    predictions = X_dask.map_blocks(
        predict_batch,
        dtype=np.int64,
        drop_axis=1,
    )

    return predictions.compute()


class ParallelDataProcessor:
    """Scalable data processing pipeline using Dask.

    This class demonstrates how to build a scalable data processing
    pipeline for sensor data that can handle larger-than-memory datasets.

    Example:
        >>> processor = ParallelDataProcessor(n_workers=4)
        >>> processor.start()
        >>> features = processor.process_files("data/*.csv")
        >>> processor.stop()
    """

    def __init__(
        self,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        memory_limit: str = "2GB",
    ) -> None:
        """Initialize the processor.

        Args:
            n_workers: Number of Dask workers.
            threads_per_worker: Threads per worker.
            memory_limit: Memory limit per worker.
        """
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.client: Client | None = None
        self.cluster: LocalCluster | None = None

    def start(self) -> None:
        """Start the Dask cluster."""
        self.cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
        )
        self.client = Client(self.cluster)
        logger.info(f"Started Dask cluster: {self.client.dashboard_link}")

    def stop(self) -> None:
        """Stop the Dask cluster."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
        logger.info("Stopped Dask cluster")

    def process_files(
        self,
        file_pattern: str,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Process multiple files in parallel.

        Args:
            file_pattern: Glob pattern for input files.
            feature_columns: Columns to process.

        Returns:
            Processed DataFrame.
        """
        if self.client is None:
            raise RuntimeError("Cluster not started. Call start() first.")

        # Load files
        ddf = dd.read_csv(file_pattern)

        if feature_columns is None:
            feature_columns = list(ddf.columns)

        # Basic aggregation
        result = ddf[feature_columns].describe().compute()
        return result

    def parallel_apply(
        self,
        df: pd.DataFrame,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        n_partitions: int | None = None,
    ) -> pd.DataFrame:
        """Apply a function in parallel.

        Args:
            df: Input DataFrame.
            func: Function to apply.
            n_partitions: Number of partitions.

        Returns:
            Processed DataFrame.
        """
        if n_partitions is None:
            n_partitions = self.n_workers

        ddf = dd.from_pandas(df, npartitions=n_partitions)
        result = ddf.map_partitions(func)
        return result.compute()


def demonstrate_dask_processing() -> None:
    """Demonstrate Dask parallel processing capabilities.

    This function shows how to use Dask for scalable data processing.
    """
    print("=" * 60)
    print("Dask Parallel Processing Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_samples = 100000
    df = pd.DataFrame({
        "acc_x": np.random.randn(n_samples),
        "acc_y": np.random.randn(n_samples),
        "acc_z": np.random.randn(n_samples),
        "gyro_x": np.random.randn(n_samples),
        "gyro_y": np.random.randn(n_samples),
        "gyro_z": np.random.randn(n_samples),
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="20ms"),
    })

    print(f"Sample data: {len(df):,} rows")

    # Convert to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=4)
    print(f"Dask partitions: {ddf.npartitions}")

    # Parallel aggregation
    print("\nParallel aggregation:")
    stats = ddf[["acc_x", "acc_y", "acc_z"]].agg(["mean", "std", "min", "max"]).compute()
    print(stats)

    # Parallel transformation
    print("\nParallel transformation (computing magnitude):")
    ddf["magnitude"] = (ddf["acc_x"]**2 + ddf["acc_y"]**2 + ddf["acc_z"]**2).apply(
        np.sqrt, meta=("magnitude", float)
    )
    mean_mag = ddf["magnitude"].mean().compute()
    print(f"Mean magnitude: {mean_mag:.4f}")

    # Feature extraction demo
    def extract_simple_features(partition: pd.DataFrame) -> pd.DataFrame:
        """Extract simple rolling features."""
        result = pd.DataFrame()
        for col in ["acc_x", "acc_y", "acc_z"]:
            result[f"{col}_rolling_mean"] = partition[col].rolling(10).mean()
            result[f"{col}_rolling_std"] = partition[col].rolling(10).std()
        return result.dropna()

    print("\nParallel feature extraction:")
    features = ddf.map_partitions(extract_simple_features).compute()
    print(f"Extracted features: {features.shape}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_dask_processing()
