"""Runs the full pipeline on a small synthetic dataset shaped like the real one."""
import json

import numpy as np
import pandas as pd

from src.train import main

CLASSES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]


def make_fake_dataset(path, n_per_class=30, n_genes=300, seed=0):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for i, c in enumerate(CLASSES):
        x = rng.gamma(2.0, 2.0, size=(n_per_class, n_genes))
        x[:, i * 10:(i + 1) * 10] += 6  # a few marker genes per class
        rows.append(x)
        labels += [c] * n_per_class
    X = np.vstack(rows)
    X[:, -5:] = 0  # some all-zero genes, like the real data
    idx = [f"sample_{i}" for i in range(len(X))]
    pd.DataFrame(X, index=idx, columns=[f"gene_{j}" for j in range(n_genes)]
                 ).to_csv(path / "data.csv")
    pd.DataFrame({"Class": labels}, index=idx).to_csv(path / "labels.csv")


def test_pipeline_runs(tmp_path):
    data_dir, results_dir = tmp_path / "data", tmp_path / "results"
    data_dir.mkdir()
    make_fake_dataset(data_dir)

    summary = main(data_dir, results_dir, k_genes=50, test_size=0.2, seed=0)

    assert summary["best_model"] in {"logistic_regression", "random_forest", "svm_rbf"}
    assert summary["logistic_regression"]["test_accuracy"] > 0.9
    for f in ["pca.png", "cv_scores.png", "top_genes.png", "metrics.json",
              "top_genes.csv", "best_model.joblib"]:
        assert (results_dir / f).exists(), f
    assert json.loads((results_dir / "metrics.json").read_text())["k_genes"] == 50
