
# nn_data_utils_datasets.py

import os
import hashlib
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

def prepare_vision_dataset(
    dataset_name: str
    , url: str
    , base_dir: Path = Path("data")
    , expected_splits: Tuple[str, ...] = ("train", "test")
    , checksum_sha256: Optional[str] = None
    , force_download: bool = False
    , cleanup_zip: bool = True
) -> Dict[str, Path]:
    """
    Downloads and prepares a vision dataset from a zip URL into a standard layout:
      base_dir/dataset_name/{train,test}/class_name/*.jpg

    Returns a dict with split names -> Path objects.
    """
    dataset_dir = base_dir / dataset_name
    zip_path = base_dir / f"{dataset_name}.zip"

    # If already prepared and not forcing, short-circuit
    if dataset_dir.is_dir() and all((dataset_dir / s).is_dir() for s in expected_splits) and not force_download:
        return {s: dataset_dir / s for s in expected_splits}

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Download
    if force_download or not zip_path.exists():
        download_file(url, zip_path)

    # Optional checksum
    if checksum_sha256 is not None:
        actual = sha256sum(zip_path)
        if actual.lower() != checksum_sha256.lower():
            raise ValueError(f"Checksum mismatch for {zip_path}: expected {checksum_sha256}, got {actual}")

    # Extract
    extract_zip(zip_path, dataset_dir)

    # Optionally clean up the zip
    if cleanup_zip and zip_path.exists():
        zip_path.unlink()

    # Validate expected splits
    missing = [s for s in expected_splits if not (dataset_dir / s).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing expected splits {missing} in {dataset_dir}. "
                                f"Check the zip structure or adjust expected_splits.")

    return {s: dataset_dir / s for s in expected_splits}

def summarize_image_folder(root: Path) -> List[Tuple[str, int, int]]:
    """
    Summarize a directory in ImageFolder layout.
    Returns a list of tuples: (dirpath, num_subdirs, num_files).
    Also prints a readable summary.
    """
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        rows.append((dirpath, len(dirnames), len([f for f in filenames if not f.startswith('.')])))

    for dp, nd, nf in rows:
        print(f"There are {nd} directories and {nf} images in '{dp}'.")
    return rows