"""
kaggle_downloader.py

Lightweight helper to download Kaggle datasets via the kaggle CLI from a Jupyter notebook or script.

Usage examples (from a notebook):
>>> from kaggle_downloader import download_kaggle_dataset
>>> download_kaggle_dataset("kshitij192/cars-image-dataset")
>>> # or provide full URL:
>>> download_kaggle_dataset("https://www.kaggle.com/datasets/kshitij192/cars-image-dataset")
>>> # specify destination directory and a path to kaggle.json
>>> download_kaggle_dataset("owner/dataset-name", dest_dir="data/my_dataset", kaggle_json_path="/path/to/kaggle.json")
"""

import os
import stat
import shutil
import subprocess
import sys
from pathlib import Path
import re
import zipfile

__all__ = ["download_kaggle_dataset"]

def _run(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        print("ERROR (stderr):", proc.stderr, file=sys.stderr)
    return proc

def _normalize_slug(input_str):
    """
    Accept either:
      - 'owner/dataset-name'
      - 'https://www.kaggle.com/datasets/owner/dataset-name'
      - 'https://kaggle.com/datasets/owner/dataset-name'
    Return owner/dataset-name (slug) and dataset-name (for naming).
    """
    if input_str is None:
        raise ValueError("No dataset slug or URL provided.")
    s = input_str.strip()
    # if full url, extract last two path components after /datasets/
    m = re.search(r"kaggle\.com/(?:datasets/)?([^/?#]+/[^/?#]+)", s)
    if m:
        slug = m.group(1)
    else:
        slug = s
    # validate slug pattern owner/dataset
    if "/" not in slug:
        raise ValueError(f"Invalid dataset slug: {slug}. Expected 'owner/dataset-name' or full Kaggle URL.")
    # dataset folder name: use last segment
    dataset_name = slug.split("/")[-1]
    return slug, dataset_name

def _ensure_kaggle_json(local_kaggle_json_path=None, cwd=Path.cwd()):
    """
    Ensure ~/.kaggle/kaggle.json exists. If local_kaggle_json_path provided, copy from there.
    Otherwise, look for kaggle.json in cwd.
    Returns path to ~/.kaggle/kaggle.json
    """
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    dest = kaggle_dir / "kaggle.json"

    if local_kaggle_json_path:
        local = Path(local_kaggle_json_path)
        if not local.exists():
            raise FileNotFoundError(f"Provided kaggle_json_path does not exist: {local}")
    else:
        local = Path(cwd) / "kaggle.json"
        if not local.exists():
            raise FileNotFoundError(f"kaggle.json not found in current directory ({cwd}). "
                                    "Provide kaggle_json_path or place kaggle.json in the notebook folder.")
    # Backup existing
    if dest.exists():
        backup = dest.with_suffix(".backup")
        print(f"Backing up existing {dest} to {backup}")
        shutil.copy2(dest, backup)
    print(f"Copying {local} -> {dest}")
    shutil.copy2(local, dest)
    # set secure permissions
    os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR)
    return dest

def download_kaggle_dataset(dataset_slug_or_url, dest_dir=None, kaggle_json_path=None, unzip=True, try_unzip_flag=True):
    """
    Download a Kaggle dataset using the kaggle CLI.

    Parameters:
      dataset_slug_or_url: str, either 'owner/dataset-name' or full Kaggle dataset URL
      dest_dir: Path or str where to extract files. If None, defaults to './kaggle_datasets/<dataset-name>'
      kaggle_json_path: optional path to kaggle.json (if not provided, expects kaggle.json in current working directory)
      unzip: bool, whether to attempt --unzip with the CLI or to extract zip after download
      try_unzip_flag: bool, whether to try using --unzip first; if False, always download zip and extract.
    Returns:
      Path to the dataset directory where files were extracted.
    """
    cwd = Path.cwd()
    slug, dataset_name = _normalize_slug(dataset_slug_or_url)
    if dest_dir is None:
        dest_dir = cwd / "kaggle_datasets" / dataset_name
    else:
        dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Ensure kaggle.json is in ~/.kaggle
    _ensure_kaggle_json(kaggle_json_path, cwd=cwd)

    # Try using kaggle CLI with --unzip (if requested)
    if unzip and try_unzip_flag:
        cmd = ["kaggle", "datasets", "download", "-d", slug, "--unzip", "-p", str(dest_dir)]
        proc = _run(cmd, cwd=cwd)
        if proc.returncode == 0:
            print("Downloaded and unzipped dataset to:", dest_dir)
            return dest_dir
        else:
            print("Warning: --unzip approach failed — falling back to download-then-unzip flow.")

    # Download zip into cwd (not dest_dir) to prevent nested folder issues
    zip_target_dir = cwd
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(zip_target_dir)]
    proc = _run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError("kaggle datasets download failed. See output above for details.")

    # Locate downloaded zip file. CLI typically names it '<dataset-name>.zip' or '<owner>-<dataset-name>.zip'
    possible = list(zip_target_dir.glob("*.zip"))
    if not possible:
        raise FileNotFoundError("Downloaded zip file not found in current directory after kaggle download.")
    # Try to pick the most relevant zip (by dataset_name)
    zip_path = None
    for p in possible:
        if dataset_name in p.name:
            zip_path = p
            break
    if zip_path is None:
        zip_path = possible[0]
    print("Found zip:", zip_path)

    # Extract into dest_dir
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=str(dest_dir))
    # Optionally remove zip
    try:
        zip_path.unlink()
    except Exception:
        pass

    print("Download and extraction complete. Dataset is at:", dest_dir)
    return dest_dir

# If run as a script, allow simple CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Kaggle datasets via kaggle CLI and extract locally.")
    parser.add_argument("dataset", help="Kaggle dataset slug (owner/dataset-name) or full URL")
    parser.add_argument("-p", "--path", help="Destination directory (default ./kaggle_datasets/<dataset>)", default=None)
    parser.add_argument("--kaggle-json", help="Path to kaggle.json (if not present in CWD)", default=None)
    parser.add_argument("--no-unzip", help="Don't try --unzip with kaggle CLI; download zip and extract instead", action="store_true")
    args = parser.parse_args()
    download_kaggle_dataset(args.dataset, dest_dir=args.path, kaggle_json_path=args.kaggle_json, unzip=not args.no_unzip)