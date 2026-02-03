"""FastAPI deployment endpoint for HAR model inference.

This module provides a REST API for activity prediction using
trained models from the MLflow registry.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global model storage
_model: Any = None
_model_info: dict[str, Any] = {}


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint.

    Attributes:
        features: List of feature values (561 features for HAR).
        return_probabilities: Whether to return class probabilities.
    """

    features: list[float] = Field(
        ...,
        description="Feature vector for prediction (561 features for HAR)",
        min_length=1,
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return class probabilities",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [0.0] * 561,
                "return_probabilities": True,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction.

    Attributes:
        samples: List of feature vectors.
        return_probabilities: Whether to return probabilities.
    """

    samples: list[list[float]] = Field(
        ...,
        description="List of feature vectors",
        min_length=1,
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return class probabilities",
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint.

    Attributes:
        prediction: Predicted activity ID.
        activity: Predicted activity name.
        probabilities: Class probabilities (if requested).
        model_version: Version of the model used.
    """

    prediction: int = Field(..., description="Predicted class ID")
    activity: str = Field(..., description="Predicted activity name")
    probabilities: dict[str, float] | None = Field(
        default=None, description="Class probabilities"
    )
    model_version: str = Field(..., description="Model version used")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 0,
                "activity": "WALKING",
                "probabilities": {
                    "WALKING": 0.85,
                    "WALKING_UPSTAIRS": 0.05,
                    "WALKING_DOWNSTAIRS": 0.05,
                    "SITTING": 0.02,
                    "STANDING": 0.02,
                    "LAYING": 0.01,
                },
                "model_version": "1.0.0",
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction.

    Attributes:
        predictions: List of prediction results.
        model_version: Model version used.
    """

    predictions: list[dict[str, Any]] = Field(
        ..., description="List of predictions"
    )
    model_version: str = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    """Response schema for health endpoint.

    Attributes:
        status: Health status.
        model_loaded: Whether model is loaded.
        model_version: Model version.
        timestamp: Current timestamp.
    """

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str | None = Field(None, description="Loaded model version")
    timestamp: str = Field(..., description="Current timestamp")


class ModelInfoResponse(BaseModel):
    """Response schema for model info endpoint.

    Attributes:
        model_name: Name of the model.
        model_version: Version number.
        model_type: Type of model (sklearn, pytorch, etc.).
        n_features: Expected number of input features.
        n_classes: Number of output classes.
        class_names: Names of activity classes.
    """

    model_name: str
    model_version: str
    model_type: str
    n_features: int
    n_classes: int
    class_names: list[str]


# Activity labels mapping
ACTIVITY_LABELS = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}


def load_model_from_path(model_path: str | Path) -> Any:
    """Load a model from a file path.

    Args:
        model_path: Path to the model file.

    Returns:
        Loaded model object.
    """
    import joblib

    model_path = Path(model_path)

    if model_path.suffix == ".joblib":
        return joblib.load(model_path)
    elif model_path.suffix in (".pt", ".pth"):
        import torch
        return torch.load(model_path)
    else:
        # Try joblib as default
        return joblib.load(model_path)


def load_model_from_mlflow(
    model_name: str = "har-classifier",
    stage: str = "Production",
) -> Any:
    """Load a model from MLflow registry.

    Args:
        model_name: Registered model name.
        stage: Model stage to load.

    Returns:
        Loaded model object.
    """
    import mlflow

    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading model from {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def initialize_model() -> None:
    """Initialize the model on startup."""
    global _model, _model_info

    # Try to load from environment variable or default path
    model_path = os.environ.get("MODEL_PATH")
    model_name = os.environ.get("MODEL_NAME", "har-classifier")
    model_stage = os.environ.get("MODEL_STAGE", "Production")

    try:
        if model_path:
            logger.info(f"Loading model from path: {model_path}")
            _model = load_model_from_path(model_path)
            _model_info = {
                "name": "local-model",
                "version": "local",
                "type": type(_model).__name__,
            }
        else:
            logger.info(f"Loading model from MLflow: {model_name}/{model_stage}")
            _model = load_model_from_mlflow(model_name, model_stage)
            _model_info = {
                "name": model_name,
                "version": model_stage,
                "type": "mlflow-pyfunc",
            }

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        logger.info("API will start without model - use /reload to load later")
        _model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting HAR Prediction API")
    initialize_model()
    yield
    # Shutdown
    logger.info("Shutting down HAR Prediction API")


# Create FastAPI application
app = FastAPI(
    title="FitTrack HAR Prediction API",
    description="REST API for Human Activity Recognition using sensor data",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns:
        Health status including model loading state.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        model_version=_model_info.get("version"),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Get information about the loaded model.

    Returns:
        Model metadata including features and classes.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    return ModelInfoResponse(
        model_name=_model_info.get("name", "unknown"),
        model_version=_model_info.get("version", "unknown"),
        model_type=_model_info.get("type", "unknown"),
        n_features=561,  # HAR dataset
        n_classes=6,
        class_names=list(ACTIVITY_LABELS.values()),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Make a single prediction.

    Args:
        request: Prediction request with features.

    Returns:
        Predicted activity and optional probabilities.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Use /reload endpoint to load a model.",
        )

    try:
        # Convert to numpy array
        features = np.array(request.features).reshape(1, -1)

        # Make prediction
        prediction = int(_model.predict(features)[0])
        activity = ACTIVITY_LABELS.get(prediction, f"UNKNOWN_{prediction}")

        # Get probabilities if requested
        probabilities = None
        if request.return_probabilities:
            if hasattr(_model, "predict_proba"):
                probs = _model.predict_proba(features)[0]
                probabilities = {
                    ACTIVITY_LABELS.get(i, f"CLASS_{i}"): float(p)
                    for i, p in enumerate(probs)
                }

        return PredictionResponse(
            prediction=prediction,
            activity=activity,
            probabilities=probabilities,
            model_version=_model_info.get("version", "unknown"),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Make batch predictions.

    Args:
        request: Batch prediction request with multiple samples.

    Returns:
        List of predictions for all samples.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        # Convert to numpy array
        features = np.array(request.samples)

        # Make predictions
        predictions = _model.predict(features)

        # Get probabilities if requested
        probs = None
        if request.return_probabilities and hasattr(_model, "predict_proba"):
            probs = _model.predict_proba(features)

        # Build response
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "prediction": int(pred),
                "activity": ACTIVITY_LABELS.get(int(pred), f"UNKNOWN_{pred}"),
            }
            if probs is not None:
                result["probabilities"] = {
                    ACTIVITY_LABELS.get(j, f"CLASS_{j}"): float(p)
                    for j, p in enumerate(probs[i])
                }
            results.append(result)

        return BatchPredictionResponse(
            predictions=results,
            model_version=_model_info.get("version", "unknown"),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.post("/reload", tags=["Model"])
async def reload_model(
    model_path: str | None = None,
    model_name: str | None = None,
    model_stage: str = "Production",
) -> dict[str, str]:
    """Reload the model.

    Args:
        model_path: Local path to model file.
        model_name: MLflow registered model name.
        model_stage: MLflow model stage.

    Returns:
        Status message.
    """
    global _model, _model_info

    try:
        if model_path:
            _model = load_model_from_path(model_path)
            _model_info = {
                "name": "local-model",
                "version": Path(model_path).name,
                "type": type(_model).__name__,
            }
        elif model_name:
            _model = load_model_from_mlflow(model_name, model_stage)
            _model_info = {
                "name": model_name,
                "version": model_stage,
                "type": "mlflow-pyfunc",
            }
        else:
            raise ValueError("Must provide either model_path or model_name")

        return {"status": "success", "message": f"Model loaded: {_model_info}"}

    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}",
        )


@app.get("/activities", tags=["Reference"])
async def list_activities() -> dict[int, str]:
    """List all activity classes.

    Returns:
        Dictionary mapping class IDs to activity names.
    """
    return ACTIVITY_LABELS


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
