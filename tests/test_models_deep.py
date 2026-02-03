"""Tests for deep learning models and data loaders."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fittrack.models.data_loaders import (
    DataModule,
    HARDataset,
    TimeSeriesDataset,
    create_data_loaders,
    create_sequence_loaders,
    reshape_for_sequence_model,
)
from fittrack.models.deep_learning import (
    ActivityCNN,
    ActivityLSTM,
    EarlyStopping,
    HARClassifier,
    TrainingConfig,
    TrainingHistory,
    create_model,
    evaluate,
    predict,
    train_epoch,
    train_model,
)


class TestHARDataset:
    """Tests for HARDataset."""

    def test_dataset_creation(self) -> None:
        """Test basic dataset creation."""
        X = np.random.randn(100, 20).astype(np.float32)
        y = np.random.randint(0, 6, 100)

        dataset = HARDataset(X, y)

        assert len(dataset) == 100
        x, label = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert x.shape == (20,)

    def test_dataset_without_labels(self) -> None:
        """Test dataset without labels (for inference)."""
        X = np.random.randn(100, 20).astype(np.float32)

        dataset = HARDataset(X, y=None)

        result = dataset[0]
        assert isinstance(result, torch.Tensor)

    def test_dataset_dtype(self) -> None:
        """Test correct dtype conversion."""
        X = np.random.randn(100, 20)  # float64
        y = np.array([0] * 100)

        dataset = HARDataset(X, y)

        x, label = dataset[0]
        assert x.dtype == torch.float32
        assert label.dtype == torch.long


class TestTimeSeriesDataset:
    """Tests for TimeSeriesDataset."""

    def test_timeseries_creation(self) -> None:
        """Test 3D time-series dataset."""
        X = np.random.randn(100, 128, 9).astype(np.float32)
        y = np.random.randint(0, 6, 100)

        dataset = TimeSeriesDataset(X, y)

        assert len(dataset) == 100
        x, label = dataset[0]
        assert x.shape == (128, 9)

    def test_invalid_dimensions(self) -> None:
        """Test that 2D input raises error."""
        X = np.random.randn(100, 20).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 3D input"):
            TimeSeriesDataset(X, np.zeros(100))


class TestDataLoaders:
    """Tests for data loader creation."""

    def test_create_data_loaders(self) -> None:
        """Test creating train/val/test loaders."""
        X_train = np.random.randn(100, 20).astype(np.float32)
        y_train = np.random.randint(0, 6, 100)
        X_val = np.random.randn(20, 20).astype(np.float32)
        y_val = np.random.randint(0, 6, 20)

        loaders = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=16
        )

        assert "train" in loaders
        assert "val" in loaders
        assert isinstance(loaders["train"], DataLoader)

    def test_weighted_sampling(self) -> None:
        """Test weighted random sampling."""
        # Create imbalanced data
        X_train = np.random.randn(100, 20).astype(np.float32)
        y_train = np.array([0] * 90 + [1] * 10)

        loaders = create_data_loaders(
            X_train, y_train, weighted_sampling=True, batch_size=16
        )

        # Sampler should be used
        assert loaders["train"].sampler is not None


class TestReshape:
    """Tests for reshape function."""

    def test_reshape_for_sequence(self) -> None:
        """Test reshaping flat features to sequence."""
        X = np.random.randn(100, 561).astype(np.float32)

        X_seq = reshape_for_sequence_model(X, 561, 1)

        assert X_seq.shape == (100, 561, 1)

    def test_reshape_multi_channel(self) -> None:
        """Test reshaping with multiple channels."""
        X = np.random.randn(100, 18).astype(np.float32)

        X_seq = reshape_for_sequence_model(X, 6, 3)

        assert X_seq.shape == (100, 6, 3)

    def test_reshape_invalid(self) -> None:
        """Test error on invalid reshape."""
        X = np.random.randn(100, 20).astype(np.float32)

        with pytest.raises(ValueError, match="Cannot reshape"):
            reshape_for_sequence_model(X, 10, 3)


class TestDataModule:
    """Tests for DataModule."""

    def test_data_module_basic(self) -> None:
        """Test basic data module."""
        X_train = np.random.randn(100, 20).astype(np.float32)
        y_train = np.random.randint(0, 6, 100)

        dm = DataModule(X_train, y_train, batch_size=16)
        dm.setup()

        assert dm.train_loader is not None
        assert dm.n_features == 20
        assert dm.n_classes == 6


class TestActivityLSTM:
    """Tests for LSTM model."""

    def test_lstm_forward(self) -> None:
        """Test LSTM forward pass."""
        model = ActivityLSTM(input_size=9, hidden_size=32, num_classes=6)
        x = torch.randn(8, 128, 9)

        output = model(x)

        assert output.shape == (8, 6)

    def test_lstm_bidirectional(self) -> None:
        """Test bidirectional LSTM."""
        model = ActivityLSTM(
            input_size=9, hidden_size=32, num_classes=6, bidirectional=True
        )
        x = torch.randn(8, 128, 9)

        output = model(x)

        assert output.shape == (8, 6)

    def test_lstm_single_layer(self) -> None:
        """Test single-layer LSTM (no dropout between layers)."""
        model = ActivityLSTM(
            input_size=9, hidden_size=32, num_layers=1, num_classes=6
        )
        x = torch.randn(8, 128, 9)

        output = model(x)

        assert output.shape == (8, 6)


class TestActivityCNN:
    """Tests for CNN model."""

    def test_cnn_forward(self) -> None:
        """Test CNN forward pass."""
        model = ActivityCNN(in_channels=9, num_classes=6, seq_length=128)
        x = torch.randn(8, 128, 9)

        output = model(x)

        assert output.shape == (8, 6)

    def test_cnn_custom_channels(self) -> None:
        """Test CNN with custom channel configuration."""
        model = ActivityCNN(
            in_channels=9,
            num_classes=6,
            channels=[32, 64],
            kernel_sizes=[5, 3],
        )
        x = torch.randn(8, 128, 9)

        output = model(x)

        assert output.shape == (8, 6)


class TestHARClassifier:
    """Tests for MLP classifier."""

    def test_mlp_forward(self) -> None:
        """Test MLP forward pass."""
        model = HARClassifier(input_size=561, num_classes=6)
        x = torch.randn(8, 561)

        output = model(x)

        assert output.shape == (8, 6)

    def test_mlp_custom_hidden(self) -> None:
        """Test MLP with custom hidden layers."""
        model = HARClassifier(
            input_size=100, num_classes=6, hidden_sizes=[64, 32]
        )
        x = torch.randn(8, 100)

        output = model(x)

        assert output.shape == (8, 6)


class TestEarlyStopping:
    """Tests for early stopping."""

    def test_early_stopping_improves(self) -> None:
        """Test that improving loss doesn't trigger stop."""
        early_stop = EarlyStopping(patience=3)
        model = HARClassifier(input_size=10, num_classes=2)

        assert not early_stop(1.0, model)
        assert not early_stop(0.9, model)
        assert not early_stop(0.8, model)

    def test_early_stopping_triggers(self) -> None:
        """Test that stagnant loss triggers stop."""
        early_stop = EarlyStopping(patience=3)
        model = HARClassifier(input_size=10, num_classes=2)

        early_stop(1.0, model)
        early_stop(1.0, model)
        early_stop(1.0, model)
        assert early_stop(1.0, model)  # Should trigger

    def test_early_stopping_counter_reset(self) -> None:
        """Test that counter resets on improvement."""
        early_stop = EarlyStopping(patience=3)
        model = HARClassifier(input_size=10, num_classes=2)

        early_stop(1.0, model)
        early_stop(1.0, model)
        assert not early_stop(0.5, model)  # Improves, reset counter
        assert early_stop.counter == 0


class TestTrainingUtils:
    """Tests for training utilities."""

    @pytest.fixture
    def simple_data(self) -> tuple:
        """Create simple training data."""
        X = np.random.randn(100, 20).astype(np.float32)
        y = np.random.randint(0, 4, 100)
        dataset = HARDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)
        return loader

    def test_train_epoch(self, simple_data: DataLoader) -> None:
        """Test training for one epoch."""
        model = HARClassifier(input_size=20, num_classes=4)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        loss, acc = train_epoch(
            model, simple_data, optimizer, criterion, torch.device("cpu")
        )

        assert loss > 0
        assert 0.0 <= acc <= 1.0

    def test_evaluate(self, simple_data: DataLoader) -> None:
        """Test model evaluation."""
        model = HARClassifier(input_size=20, num_classes=4)
        criterion = torch.nn.CrossEntropyLoss()

        loss, acc = evaluate(model, simple_data, criterion, torch.device("cpu"))

        assert loss > 0
        assert 0.0 <= acc <= 1.0


class TestTrainModel:
    """Tests for full training loop."""

    def test_train_model_basic(self) -> None:
        """Test basic model training."""
        X_train = np.random.randn(100, 20).astype(np.float32)
        y_train = np.random.randint(0, 4, 100)
        X_val = np.random.randn(20, 20).astype(np.float32)
        y_val = np.random.randint(0, 4, 20)

        train_loader = DataLoader(HARDataset(X_train, y_train), batch_size=16)
        val_loader = DataLoader(HARDataset(X_val, y_val), batch_size=16)

        model = HARClassifier(input_size=20, num_classes=4)
        config = TrainingConfig(epochs=3, patience=10)

        history = train_model(
            model, train_loader, val_loader, config, device=torch.device("cpu")
        )

        assert isinstance(history, TrainingHistory)
        assert len(history.train_losses) == 3
        assert len(history.val_losses) == 3


class TestPredict:
    """Tests for prediction."""

    def test_predict(self) -> None:
        """Test prediction generation."""
        X = np.random.randn(50, 20).astype(np.float32)
        y = np.random.randint(0, 4, 50)

        loader = DataLoader(HARDataset(X, y), batch_size=16)
        model = HARClassifier(input_size=20, num_classes=4)

        preds, probs = predict(model, loader, torch.device("cpu"), return_probs=True)

        assert preds.shape == (50,)
        assert probs.shape == (50, 4)
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestCreateModel:
    """Tests for model factory."""

    def test_create_lstm(self) -> None:
        """Test creating LSTM model."""
        model = create_model("lstm", input_size=9, num_classes=6)
        assert isinstance(model, ActivityLSTM)

    def test_create_cnn(self) -> None:
        """Test creating CNN model."""
        model = create_model("cnn", input_size=9, num_classes=6)
        assert isinstance(model, ActivityCNN)

    def test_create_mlp(self) -> None:
        """Test creating MLP model."""
        model = create_model("mlp", input_size=100, num_classes=6)
        assert isinstance(model, HARClassifier)

    def test_invalid_model_type(self) -> None:
        """Test error on invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("transformer", input_size=100, num_classes=6)  # type: ignore
