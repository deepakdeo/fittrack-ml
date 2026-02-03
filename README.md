# FitTrack ML

[![CI](https://github.com/deepakdeo/fittrack-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/deepakdeo/fittrack-ml/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end ML pipeline for fitness activity recognition from wearable sensor data. Classifies physical activities (walking, running, sitting, etc.) from accelerometer and gyroscope readings.

## Features

- **Time-series classification** using both classical ML (Random Forest, XGBoost) and deep learning (LSTM, 1D-CNN)
- **MLOps best practices** with MLflow experiment tracking and model registry
- **A/B testing framework** for statistical model comparison
- **Production-ready API** with FastAPI
- **Scalable data processing** with Dask for larger-than-memory datasets

## Quick Start

```bash
# Clone the repository
git clone https://github.com/deepakdeo/fittrack-ml.git
cd fittrack-ml

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Download the dataset
fittrack-download

# Run tests
pytest
```

## Project Structure

```
fittrack-ml/
├── src/fittrack/
│   ├── data/           # Data ingestion and preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # ML model implementations
│   ├── mlops/          # MLflow tracking and A/B testing
│   └── deployment/     # FastAPI endpoint
├── notebooks/          # Jupyter notebooks for EDA and experiments
├── tests/              # Unit and integration tests
├── data/               # Raw and processed data (gitignored)
└── docs/               # Documentation and figures
```

## Dataset

This project uses the [UCI Human Activity Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones):

- **10,299 samples** from 30 subjects
- **6 activity classes**: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
- **561 features** derived from accelerometer and gyroscope signals

## License

MIT License - see [LICENSE](LICENSE) for details.
