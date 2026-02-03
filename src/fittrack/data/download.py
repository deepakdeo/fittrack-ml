"""Download and extract the UCI HAR dataset.

The UCI Human Activity Recognition dataset contains sensor data from 30 subjects
performing 6 activities: Walking, Walking Upstairs, Walking Downstairs, Sitting,
Standing, and Laying.

Dataset source: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
"""

import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

UCI_HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"


def download_har_dataset(
    url: str = UCI_HAR_URL,
    data_dir: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Download and extract the UCI HAR dataset.

    Args:
        url: URL to download the dataset from.
        data_dir: Directory to save the dataset. Defaults to data/raw/.
        force: If True, re-download even if data already exists.

    Returns:
        Path to the extracted dataset directory.

    Raises:
        requests.RequestException: If download fails.
        zipfile.BadZipFile: If the downloaded file is corrupted.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = data_dir / "UCI HAR Dataset"
    zip_path = data_dir / "uci_har_dataset.zip"

    # Check if already downloaded
    if extracted_dir.exists() and not force:
        logger.info(f"Dataset already exists at {extracted_dir}")
        return extracted_dir

    # Download with progress bar
    logger.info(f"Downloading UCI HAR dataset from {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    # Extract
    logger.info(f"Extracting dataset to {data_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Clean up zip file
    zip_path.unlink()
    logger.info(f"Dataset extracted to {extracted_dir}")

    return extracted_dir


def main() -> None:
    """CLI entry point for downloading the dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    path = download_har_dataset()
    print(f"Dataset downloaded and extracted to: {path}")


if __name__ == "__main__":
    main()
