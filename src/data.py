"""Loading and basic checks for the expression matrix."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir: Path = DATA_DIR):
    """Return (X, y). X is samples x genes, y is the cancer type label."""
    data_dir = Path(data_dir)
    X = pd.read_csv(data_dir / "data.csv", index_col=0)
    y = pd.read_csv(data_dir / "labels.csv", index_col=0)["Class"]
    y = y.loc[X.index]
    return X.astype(np.float32), y


def summarize(X: pd.DataFrame, y: pd.Series) -> None:
    print(f"Samples: {X.shape[0]}   Genes: {X.shape[1]}")
    print(f"Expression range: {X.values.min():.2f} to {X.values.max():.2f}")
    print(f"Genes with zero expression in every sample: {int((X == 0).all().sum())}")
    print("Samples per cancer type:")
    print(y.value_counts().to_string())
