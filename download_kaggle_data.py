import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset(dataset_name: str, download_path: str = "data"):
    """
    Downloads and extracts a Kaggle dataset.

    Parameters:
        dataset_name (str): The Kaggle dataset identifier (e.g., "username/dataset-name").
        download_path (str): Local path to save the dataset.
    """
    os.makedirs(download_path, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset {dataset_name}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset downloaded and extracted to {download_path}/")

if __name__ == "__main__":
    DATASET_NAME = "example/dataset-name"  # Replace with your Kaggle dataset identifier
    DOWNLOAD_PATH = "data"  # Path to save the dataset
    download_kaggle_dataset(DATASET_NAME, DOWNLOAD_PATH)
