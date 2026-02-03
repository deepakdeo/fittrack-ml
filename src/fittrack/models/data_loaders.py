"""PyTorch data utilities for activity recognition.

This module provides Dataset classes and DataLoader factories for
training deep learning models on HAR data.
"""

import logging
from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class HARDataset(Dataset):
    """PyTorch Dataset for Human Activity Recognition data.

    Supports both pre-computed features (2D) and raw time-series (3D) data.

    Example:
        >>> X = np.random.randn(1000, 561)  # Pre-computed features
        >>> y = np.random.randint(0, 6, 1000)
        >>> dataset = HARDataset(X, y)
        >>> features, label = dataset[0]
        >>> print(features.shape)  # torch.Size([561])
    """

    def __init__(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer] | None = None,
        transform: Callable | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            X: Feature array of shape (n_samples, n_features) or
               (n_samples, seq_len, n_channels) for time-series.
            y: Label array of shape (n_samples,). None for inference.
            transform: Optional transform to apply to samples.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, label) if labels exist, else just features.
        """
        x = self.X[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.y is not None:
            return x, self.y[idx]
        return x


class TimeSeriesDataset(Dataset):
    """Dataset for raw time-series sensor data.

    Expects 3D input of shape (n_samples, sequence_length, n_channels).

    Example:
        >>> X = np.random.randn(1000, 128, 9)  # 128 timesteps, 9 sensor channels
        >>> y = np.random.randint(0, 6, 1000)
        >>> dataset = TimeSeriesDataset(X, y)
        >>> x, label = dataset[0]
        >>> print(x.shape)  # torch.Size([128, 9])
    """

    def __init__(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer] | None = None,
        augment: bool = False,
        noise_std: float = 0.01,
    ) -> None:
        """Initialize the time-series dataset.

        Args:
            X: Time-series array of shape (n_samples, seq_len, n_channels).
            y: Label array of shape (n_samples,).
            augment: Whether to apply random augmentation during training.
            noise_std: Standard deviation of Gaussian noise for augmentation.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {X.shape}")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Get a sample with optional augmentation."""
        x = self.X[idx].clone()

        if self.augment and self.training:
            # Add Gaussian noise
            x = x + torch.randn_like(x) * self.noise_std

            # Random time shift
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(-5, 6, (1,)).item()
                x = torch.roll(x, shifts=int(shift), dims=0)

        if self.y is not None:
            return x, self.y[idx]
        return x

    @property
    def training(self) -> bool:
        """Check if in training mode."""
        return self.augment


def create_data_loaders(
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    X_val: NDArray[np.floating] | None = None,
    y_val: NDArray[np.integer] | None = None,
    X_test: NDArray[np.floating] | None = None,
    y_test: NDArray[np.integer] | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    weighted_sampling: bool = False,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Create DataLoaders for training, validation, and test sets.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        X_test: Test features (optional).
        y_test: Test labels (optional).
        batch_size: Batch size for training and evaluation.
        num_workers: Number of data loading workers.
        weighted_sampling: Use weighted sampling for class imbalance.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders.

    Example:
        >>> loaders = create_data_loaders(X_train, y_train, X_val, y_val)
        >>> for batch_x, batch_y in loaders['train']:
        ...     # Training loop
        ...     pass
    """
    loaders = {}

    # Training loader
    train_dataset = HARDataset(X_train, y_train)

    if weighted_sampling:
        # Compute sample weights for balanced sampling
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Validation loader
    if X_val is not None and y_val is not None:
        val_dataset = HARDataset(X_val, y_val)
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Test loader
    if X_test is not None and y_test is not None:
        test_dataset = HARDataset(X_test, y_test)
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    logger.info(
        f"Created DataLoaders - Train: {len(train_dataset)} samples, batch_size={batch_size}"
    )

    return loaders


def create_sequence_loaders(
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    X_val: NDArray[np.floating] | None = None,
    y_val: NDArray[np.integer] | None = None,
    X_test: NDArray[np.floating] | None = None,
    y_test: NDArray[np.integer] | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    augment_train: bool = True,
) -> dict[str, DataLoader]:
    """Create DataLoaders for sequence/time-series data.

    Args:
        X_train: Training sequences of shape (n_samples, seq_len, n_channels).
        y_train: Training labels.
        X_val: Validation sequences (optional).
        y_val: Validation labels (optional).
        X_test: Test sequences (optional).
        y_test: Test labels (optional).
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        augment_train: Whether to augment training data.

    Returns:
        Dictionary with DataLoaders.
    """
    loaders = {}

    train_dataset = TimeSeriesDataset(X_train, y_train, augment=augment_train)
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    if X_val is not None and y_val is not None:
        val_dataset = TimeSeriesDataset(X_val, y_val, augment=False)
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    if X_test is not None and y_test is not None:
        test_dataset = TimeSeriesDataset(X_test, y_test, augment=False)
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders


def reshape_for_sequence_model(
    X: NDArray[np.floating],
    sequence_length: int,
    n_channels: int,
) -> NDArray[np.floating]:
    """Reshape flat features into sequence format for LSTM/CNN.

    Useful when you have pre-computed features that you want to treat
    as a sequence for temporal modeling.

    Args:
        X: Feature array of shape (n_samples, n_features).
        sequence_length: Desired sequence length.
        n_channels: Number of channels (features per timestep).

    Returns:
        Reshaped array of shape (n_samples, sequence_length, n_channels).

    Raises:
        ValueError: If n_features != sequence_length * n_channels.

    Example:
        >>> X = np.random.randn(100, 561)
        >>> X_seq = reshape_for_sequence_model(X, 561, 1)
        >>> X_seq.shape
        (100, 561, 1)
    """
    n_samples, n_features = X.shape
    expected_features = sequence_length * n_channels

    if n_features != expected_features:
        raise ValueError(
            f"Cannot reshape {n_features} features into "
            f"({sequence_length}, {n_channels}). Need {expected_features} features."
        )

    return X.reshape(n_samples, sequence_length, n_channels)


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


class DataModule:
    """High-level data module for managing datasets and loaders.

    Example:
        >>> dm = DataModule(X_train, y_train, X_val, y_val, X_test, y_test)
        >>> dm.setup()
        >>> for x, y in dm.train_loader:
        ...     # Training loop
        ...     pass
    """

    def __init__(
        self,
        X_train: NDArray[np.floating],
        y_train: NDArray[np.integer],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.integer] | None = None,
        X_test: NDArray[np.floating] | None = None,
        y_test: NDArray[np.integer] | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        is_sequence: bool = False,
    ) -> None:
        """Initialize the data module.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            X_test: Test features.
            y_test: Test labels.
            batch_size: Batch size for all loaders.
            num_workers: Number of data loading workers.
            is_sequence: Whether data is sequential (3D).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_sequence = is_sequence

        self._loaders: dict[str, DataLoader] | None = None

    def setup(self) -> None:
        """Set up DataLoaders."""
        if self.is_sequence:
            self._loaders = create_sequence_loaders(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                self.X_test,
                self.y_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        else:
            self._loaders = create_data_loaders(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                self.X_test,
                self.y_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

    @property
    def train_loader(self) -> DataLoader:
        """Get training DataLoader."""
        if self._loaders is None:
            self.setup()
        return self._loaders["train"]  # type: ignore

    @property
    def val_loader(self) -> DataLoader | None:
        """Get validation DataLoader."""
        if self._loaders is None:
            self.setup()
        return self._loaders.get("val")  # type: ignore

    @property
    def test_loader(self) -> DataLoader | None:
        """Get test DataLoader."""
        if self._loaders is None:
            self.setup()
        return self._loaders.get("test")  # type: ignore

    @property
    def n_features(self) -> int:
        """Get number of features."""
        if self.is_sequence:
            return self.X_train.shape[2]  # n_channels
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        """Get number of classes."""
        return len(np.unique(self.y_train))

    @property
    def sequence_length(self) -> int | None:
        """Get sequence length for sequential data."""
        if self.is_sequence:
            return self.X_train.shape[1]
        return None
