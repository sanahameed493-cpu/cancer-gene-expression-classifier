"""Model pipelines.

Every preprocessing step lives inside the sklearn Pipeline, so gene filtering,
scaling and feature selection are fit only on the training folds. Selecting
genes on the full dataset before cross-validation leaks test information into
training and is a common cause of inflated accuracy in expression studies.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _preprocessing(k_genes: int):
    return [
        ("drop_constant", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_genes)),
    ]


def build_models(k_genes: int = 500, random_state: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline(_preprocessing(k_genes) + [
            ("clf", LogisticRegression(C=0.1, max_iter=5000,
                                       random_state=random_state)),
        ]),
        "random_forest": Pipeline(_preprocessing(k_genes) + [
            ("clf", RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                           random_state=random_state)),
        ]),
        "svm_rbf": Pipeline(_preprocessing(k_genes) + [
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        random_state=random_state)),
        ]),
    }
