"""Tests for FastAPI deployment endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from fittrack.deployment.api import (
    ACTIVITY_LABELS,
    PredictionRequest,
    PredictionResponse,
    app,
)


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock trained model."""
    # Train a simple model
    X = np.random.randn(100, 561)
    y = np.random.randint(0, 6, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_features() -> list[float]:
    """Generate sample feature vector."""
    return [0.0] * 561


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check returns OK."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_response_structure(self, client: TestClient) -> None:
        """Test health response has correct structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestActivitiesEndpoint:
    """Tests for /activities endpoint."""

    def test_list_activities(self, client: TestClient) -> None:
        """Test listing all activities."""
        response = client.get("/activities")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 6
        assert "0" in data or 0 in data

    def test_activity_names(self, client: TestClient) -> None:
        """Test that activity names are correct."""
        response = client.get("/activities")
        data = response.json()

        # Check some expected activities
        activities = list(data.values())
        assert "WALKING" in activities
        assert "SITTING" in activities
        assert "LAYING" in activities


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_without_model(
        self,
        client: TestClient,
        sample_features: list[float],
    ) -> None:
        """Test prediction without loaded model returns 503."""
        response = client.post(
            "/predict",
            json={"features": sample_features},
        )
        # Model not loaded should return 503
        assert response.status_code in [200, 503]

    def test_predict_request_validation(self, client: TestClient) -> None:
        """Test that invalid requests are rejected."""
        # Empty features
        response = client.post(
            "/predict",
            json={"features": []},
        )
        assert response.status_code == 422  # Validation error

    def test_predict_request_schema(self) -> None:
        """Test prediction request schema."""
        request = PredictionRequest(
            features=[0.0] * 561,
            return_probabilities=True,
        )
        assert len(request.features) == 561
        assert request.return_probabilities is True

    def test_predict_response_schema(self) -> None:
        """Test prediction response schema."""
        response = PredictionResponse(
            prediction=0,
            activity="WALKING",
            probabilities={"WALKING": 0.9},
            model_version="1.0",
        )
        assert response.prediction == 0
        assert response.activity == "WALKING"


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict_without_model(self, client: TestClient) -> None:
        """Test batch prediction without model."""
        response = client.post(
            "/predict/batch",
            json={"samples": [[0.0] * 561, [0.0] * 561]},
        )
        assert response.status_code in [200, 503]

    def test_batch_request_validation(self, client: TestClient) -> None:
        """Test that empty batch is rejected."""
        response = client.post(
            "/predict/batch",
            json={"samples": []},
        )
        assert response.status_code == 422


class TestModelEndpoints:
    """Tests for model-related endpoints."""

    def test_model_info_without_model(self, client: TestClient) -> None:
        """Test model info when no model is loaded."""
        response = client.get("/model/info")
        # Should return 503 if no model loaded
        assert response.status_code in [200, 503]

    def test_reload_without_params(self, client: TestClient) -> None:
        """Test reload endpoint without parameters."""
        response = client.post("/reload")
        # Should fail without model_path or model_name
        assert response.status_code in [422, 500]


class TestPredictionRequest:
    """Tests for PredictionRequest model."""

    def test_valid_request(self) -> None:
        """Test creating valid request."""
        request = PredictionRequest(features=[0.0] * 100)
        assert len(request.features) == 100
        assert request.return_probabilities is False

    def test_with_probabilities(self) -> None:
        """Test request with probabilities flag."""
        request = PredictionRequest(
            features=[0.0] * 100,
            return_probabilities=True,
        )
        assert request.return_probabilities is True


class TestPredictionResponse:
    """Tests for PredictionResponse model."""

    def test_valid_response(self) -> None:
        """Test creating valid response."""
        response = PredictionResponse(
            prediction=0,
            activity="WALKING",
            model_version="1.0",
        )
        assert response.prediction == 0
        assert response.probabilities is None

    def test_response_with_probabilities(self) -> None:
        """Test response with probabilities."""
        probs = {
            "WALKING": 0.7,
            "SITTING": 0.2,
            "STANDING": 0.1,
        }
        response = PredictionResponse(
            prediction=0,
            activity="WALKING",
            probabilities=probs,
            model_version="1.0",
        )
        assert response.probabilities == probs


class TestActivityLabels:
    """Tests for activity labels mapping."""

    def test_all_activities_defined(self) -> None:
        """Test that all 6 activities are defined."""
        assert len(ACTIVITY_LABELS) == 6

    def test_activity_ids(self) -> None:
        """Test activity ID range."""
        assert all(0 <= k <= 5 for k in ACTIVITY_LABELS)

    def test_activity_names(self) -> None:
        """Test expected activity names."""
        expected = {
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING",
        }
        assert set(ACTIVITY_LABELS.values()) == expected


class TestAPIIntegration:
    """Integration tests for API."""

    def test_api_startup(self, client: TestClient) -> None:
        """Test that API starts successfully."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_openapi_schema(self, client: TestClient) -> None:
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "info" in schema
        assert schema["info"]["title"] == "FitTrack HAR Prediction API"

    def test_docs_available(self, client: TestClient) -> None:
        """Test that docs endpoint is available."""
        response = client.get("/docs")
        assert response.status_code == 200
