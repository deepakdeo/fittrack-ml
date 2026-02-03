"""Deep learning models for activity recognition.

This module provides LSTM and CNN models for time-series classification,
along with training utilities including early stopping and learning rate scheduling.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from fittrack.models.data_loaders import get_device

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization strength.
        patience: Early stopping patience (epochs without improvement).
        min_delta: Minimum improvement to reset patience.
        scheduler: Learning rate scheduler type.
        scheduler_patience: Patience for ReduceLROnPlateau.
        scheduler_factor: Factor for ReduceLROnPlateau.
        gradient_clip: Maximum gradient norm for clipping.
    """

    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 10
    min_delta: float = 0.001
    scheduler: Literal["plateau", "cosine", "none"] = "plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    gradient_clip: float | None = 1.0


@dataclass
class TrainingHistory:
    """Container for training history.

    Attributes:
        train_losses: Loss per epoch.
        val_losses: Validation loss per epoch.
        train_accuracies: Training accuracy per epoch.
        val_accuracies: Validation accuracy per epoch.
        learning_rates: Learning rate per epoch.
        best_epoch: Epoch with best validation loss.
        best_val_loss: Best validation loss achieved.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


class ActivityLSTM(nn.Module):
    """LSTM model for activity classification.

    Architecture:
        Input → LSTM layers → Dropout → FC → Output

    Example:
        >>> model = ActivityLSTM(input_size=561, hidden_size=128, num_classes=6)
        >>> x = torch.randn(32, 100, 561)  # (batch, seq_len, features)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 6])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        """Initialize the LSTM model.

        Args:
            input_size: Number of input features per timestep.
            hidden_size: LSTM hidden state size.
            num_layers: Number of LSTM layers.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            bidirectional: Use bidirectional LSTM.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        # Output size depends on bidirectional setting
        lstm_output_size = hidden_size * self.num_directions

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Output logits of shape (batch, num_classes).
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2, :, :]  # Last forward layer
            h_backward = h_n[-1, :, :]  # Last backward layer
            hidden = torch.cat((h_forward, h_backward), dim=1)
        else:
            hidden = h_n[-1, :, :]

        # Classification
        hidden = self.dropout(hidden)
        output = self.fc(hidden)

        return output


class ActivityCNN(nn.Module):
    """1D CNN model for time-series activity classification.

    Architecture:
        Input → Conv blocks → Global pooling → FC → Output

    Example:
        >>> model = ActivityCNN(in_channels=9, num_classes=6, seq_length=128)
        >>> x = torch.randn(32, 128, 9)  # (batch, seq_len, channels)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 6])
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 6,
        seq_length: int = 128,  # noqa: ARG002
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the CNN model.

        Args:
            in_channels: Number of input channels (sensor axes).
            num_classes: Number of output classes.
            seq_length: Input sequence length (kept for API compatibility).
            channels: List of channel sizes for conv layers.
            kernel_sizes: List of kernel sizes for conv layers.
            dropout: Dropout rate.
        """
        super().__init__()

        if channels is None:
            channels = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        self.conv_blocks = nn.ModuleList()
        prev_channels = in_channels

        for ch, ks in zip(channels, kernel_sizes, strict=False):
            block = nn.Sequential(
                nn.Conv1d(prev_channels, ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            )
            self.conv_blocks.append(block)
            prev_channels = ch

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, in_channels).
               Will be transposed to (batch, in_channels, seq_len) for Conv1d.

        Returns:
            Output logits of shape (batch, num_classes).
        """
        # Transpose for Conv1d: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)

        # Classification
        output = self.classifier(x)

        return output


class HARClassifier(nn.Module):
    """Simple feedforward classifier for pre-computed features.

    For when you have feature vectors (not sequences).

    Example:
        >>> model = HARClassifier(input_size=561, num_classes=6)
        >>> x = torch.randn(32, 561)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 6])
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 6,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the classifier.

        Args:
            input_size: Number of input features.
            num_classes: Number of output classes.
            hidden_sizes: List of hidden layer sizes.
            dropout: Dropout rate.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Example:
        >>> early_stop = EarlyStopping(patience=10)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stop(val_loss, model):
        ...         print("Early stopping!")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        restore_best: bool = True,
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum improvement to reset patience.
            restore_best: Whether to restore best model weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state: dict[str, Any] | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.
            model: Model to potentially save/restore.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best and self.best_state is not None:
                model.load_state_dict(self.best_state)
                logger.info("Restored best model weights")
            return True

        return False


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float | None = None,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device for computation.
        gradient_clip: Maximum gradient norm.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        data_loader: Data loader.
        criterion: Loss function.
        device: Device for computation.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    config: TrainingConfig | None = None,
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None,
    checkpoint_dir: Path | str | None = None,
) -> TrainingHistory:
    """Train a model with full training loop.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader (optional).
        config: Training configuration.
        class_weights: Class weights for imbalanced data.
        device: Device for computation. Auto-detected if None.
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        TrainingHistory with training metrics.

    Example:
        >>> model = ActivityLSTM(input_size=9, num_classes=6)
        >>> history = train_model(model, train_loader, val_loader)
        >>> print(f"Best val accuracy: {max(history.val_accuracies):.4f}")
    """
    if config is None:
        config = TrainingConfig()

    if device is None:
        device = get_device()

    model = model.to(device)

    # Loss function with optional class weights
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    scheduler = None
    if config.scheduler == "plateau" and val_loader is not None:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
        )
    elif config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
    )

    # Checkpoint directory
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = TrainingHistory()

    logger.info(f"Starting training for {config.epochs} epochs on {device}")

    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, config.gradient_clip
        )

        history.train_losses.append(train_loss)
        history.train_accuracies.append(train_acc)
        history.learning_rates.append(optimizer.param_groups[0]["lr"])

        # Validate
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            history.val_losses.append(val_loss)
            history.val_accuracies.append(val_acc)

            # Track best
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch

                # Save checkpoint
                if checkpoint_dir:
                    torch.save(
                        model.state_dict(),
                        checkpoint_dir / "best_model.pt",
                    )

            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping
            if early_stopping(val_loss, model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
        else:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )

    logger.info(
        f"Training complete. Best epoch: {history.best_epoch + 1}, "
        f"Best val loss: {history.best_val_loss:.4f}"
    )

    return history


@torch.no_grad()
def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
    return_probs: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Generate predictions from a model.

    Args:
        model: Trained model.
        data_loader: Data loader for prediction.
        device: Device for computation.
        return_probs: Whether to return probabilities.

    Returns:
        Tuple of (predictions, probabilities) where probabilities is None if not requested.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []

    for batch_x, _ in data_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)

        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        if return_probs:
            all_probs.append(probs.cpu().numpy())

    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs) if return_probs else None

    return predictions, probabilities


def create_model(
    model_type: Literal["lstm", "cnn", "mlp"],
    input_size: int,
    num_classes: int,
    **kwargs: Any,
) -> nn.Module:
    """Factory function to create models.

    Args:
        model_type: Type of model ("lstm", "cnn", "mlp").
        input_size: Input feature size.
        num_classes: Number of output classes.
        **kwargs: Additional model-specific arguments.

    Returns:
        Instantiated model.

    Example:
        >>> model = create_model("lstm", input_size=9, num_classes=6, hidden_size=128)
    """
    if model_type == "lstm":
        return ActivityLSTM(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type == "cnn":
        return ActivityCNN(in_channels=input_size, num_classes=num_classes, **kwargs)
    elif model_type == "mlp":
        return HARClassifier(input_size=input_size, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
