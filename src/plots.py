"""Figures saved to results/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def plot_pca(X: pd.DataFrame, y: pd.Series, out: Path) -> None:
    Xs = StandardScaler().fit_transform(X.loc[:, X.std() > 0])
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label in sorted(y.unique()):
        m = (y == label).values
        ax.scatter(pcs[m, 0], pcs[m, 1], s=14, alpha=0.75, label=label)
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% of variance)")
    ax.set_title("PCA of tumor samples (all genes)")
    ax.legend(title="Cancer type")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, labels, title: str, out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cv_scores(cv_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=cv_df, x="model", y="f1_macro", ax=ax, color="#9ecae1")
    sns.stripplot(data=cv_df, x="model", y="f1_macro", ax=ax,
                  color="black", size=4)
    ax.set_ylabel("Macro F1 (5-fold CV)")
    ax.set_xlabel("")
    ax.set_title("Cross-validation performance")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_top_genes(importances: pd.Series, out: Path, n: int = 20) -> None:
    top = importances.sort_values(ascending=False).head(n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color="#3182bd")
    ax.set_xlabel("Random forest importance")
    ax.set_title(f"Top {n} genes separating the cancer types")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
