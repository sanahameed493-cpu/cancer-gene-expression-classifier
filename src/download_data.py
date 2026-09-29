"""Download the TCGA pan-cancer RNA-seq dataset from the UCI ML Repository.

Dataset: "Gene expression cancer RNA-Seq" (UCI id 401)
801 tumor samples x 20,531 genes, 5 cancer types (BRCA, KIRC, COAD, LUAD, PRAD).

Usage:
    python -m src.download_data
"""
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _extract_all(path: Path, dest: Path) -> None:
    """Extract a zip or tar.gz, then extract any archives found inside it."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            tf.extractall(dest)
    # UCI wraps a .tar.gz inside the .zip, so unpack nested archives too
    for inner in list(dest.rglob("*.tar.gz")) + list(dest.rglob("*.tgz")):
        with tarfile.open(inner) as tf:
            tf.extractall(inner.parent)
        inner.unlink()


def download(force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    data_csv = DATA_DIR / "data.csv"
    labels_csv = DATA_DIR / "labels.csv"
    if data_csv.exists() and labels_csv.exists() and not force:
        print(f"Data already present in {DATA_DIR}")
        return DATA_DIR

    tmp = DATA_DIR / "_download"
    tmp.mkdir(exist_ok=True)
    archive = tmp / "dataset.zip"
    print(f"Downloading {URL} (about 70 MB) ...")
    urllib.request.urlretrieve(URL, archive)
    print("Extracting ...")
    _extract_all(archive, tmp)

    found_data = next(tmp.rglob("data.csv"), None)
    found_labels = next(tmp.rglob("labels.csv"), None)
    if found_data is None or found_labels is None:
        raise FileNotFoundError(
            "Could not find data.csv / labels.csv in the download. "
            "Download manually from https://archive.ics.uci.edu/dataset/401 "
            f"and place both files in {DATA_DIR}"
        )
    shutil.move(str(found_data), data_csv)
    shutil.move(str(found_labels), labels_csv)
    shutil.rmtree(tmp)
    print(f"Saved data.csv and labels.csv to {DATA_DIR}")
    return DATA_DIR


if __name__ == "__main__":
    download()
