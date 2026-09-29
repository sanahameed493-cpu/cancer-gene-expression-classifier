"""Train and evaluate cancer type classifiers.

Usage:
    python -m src.train                  # defaults: 500 genes, 20% test set
    python -m src.train --k-genes 100
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .data import DATA_DIR, load_data, summarize
from .models import build_models
from .plots import plot_confusion, plot_cv_scores, plot_pca, plot_top_genes

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def selected_gene_names(pipeline, gene_names: np.ndarray) -> np.ndarray:
    """Map the pipeline's selected features back to gene names."""
    kept = gene_names[pipeline.named_steps["drop_constant"].get_support()]
    return kept[pipeline.named_steps["select"].get_support()]


def main(data_dir: Path, results_dir: Path, k_genes: int,
         test_size: float, seed: int) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_dir)
    summarize(X, y)
    labels = sorted(y.unique())
    plot_pca(X, y, results_dir / "pca.png")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    print(f"\nTrain: {len(X_train)} samples   Test: {len(X_test)} samples")

    models = build_models(k_genes=k_genes, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    cv_rows, summary = [], {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        scores = cross_validate(model, X_train, y_train, cv=cv,
                                scoring=["accuracy", "f1_macro"], n_jobs=1)
        for fold, (acc, f1) in enumerate(zip(scores["test_accuracy"],
                                             scores["test_f1_macro"])):
            cv_rows.append({"model": name, "fold": fold,
                            "accuracy": acc, "f1_macro": f1})
        print(f"CV accuracy {scores['test_accuracy'].mean():.4f} "
              f"(+/- {scores['test_accuracy'].std():.4f}), "
              f"macro F1 {scores['test_f1_macro'].mean():.4f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Held-out test accuracy {test_acc:.4f}, macro F1 {test_f1:.4f}")
        print(classification_report(y_test, y_pred, digits=3))

        plot_confusion(y_test, y_pred, labels,
                       f"{name} (test accuracy {test_acc:.3f})",
                       results_dir / f"confusion_{name}.png")
        summary[name] = {
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1),
        }

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(results_dir / "cv_scores.csv", index=False)
    plot_cv_scores(cv_df, results_dir / "cv_scores.png")

    # Gene importance from the random forest
    rf = models["random_forest"]
    genes = selected_gene_names(rf, X.columns.to_numpy())
    importances = pd.Series(rf.named_steps["clf"].feature_importances_,
                            index=genes, name="importance")
    importances.sort_values(ascending=False).to_csv(results_dir / "top_genes.csv")
    plot_top_genes(importances, results_dir / "top_genes.png")

    best = max(summary, key=lambda m: summary[m]["cv_f1_macro_mean"])
    joblib.dump(models[best], results_dir / "best_model.joblib")
    summary["best_model"] = best
    summary["k_genes"] = k_genes
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest model by CV macro F1: {best}")
    print(f"Figures and metrics saved to {results_dir}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--k-genes", type=int, default=500,
                   help="number of genes kept by ANOVA F-test selection")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.data_dir, a.results_dir, a.k_genes, a.test_size, a.seed)
