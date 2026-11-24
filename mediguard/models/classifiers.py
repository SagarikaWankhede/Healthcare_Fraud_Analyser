"""Candidate provider-fraud classifiers.

Every model is an sklearn ``Pipeline`` whose first step is a median imputer, so imputation is
fitted on training rows only and travels with the model into inference. Class imbalance (~9% of
providers are fraud-labelled) is handled with class weights rather than resampling:

* Logistic Regression - ``class_weight='balanced'`` on signed-log1p + standardised features.
* Random Forest       - ``class_weight='balanced_subsample'``.
* AdaBoost            - balanced ``sample_weight`` passed at fit time.
* XGBoost             - ``scale_pos_weight = negatives / positives`` of the training fold.

Weighting distorts raw scores, so probabilities are calibrated afterwards (see ``evaluation``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


@dataclass
class ModelSpec:
    key: str
    name: str
    grid: dict = field(default_factory=dict)

    def configs(self) -> list[dict]:
        keys = list(self.grid)
        return [dict(zip(keys, values)) for values in product(*self.grid.values())] or [{}]

    def build(self, params: dict, y_train: np.ndarray, random_state: int = 42) -> Pipeline:
        pos = max(int((y_train == 1).sum()), 1)
        neg = int((y_train == 0).sum())
        imputer = ("impute", SimpleImputer(strategy="median"))
        if self.key == "logistic_regression":
            return Pipeline([imputer, ("log", FunctionTransformer(signed_log1p)), ("scale", StandardScaler()),
                             ("clf", LogisticRegression(class_weight="balanced", max_iter=5000,
                                                        C=params.get("C", 1.0), random_state=random_state))])
        if self.key == "random_forest":
            return Pipeline([imputer, ("clf", RandomForestClassifier(
                n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
                max_depth=params.get("max_depth"), min_samples_leaf=params.get("min_samples_leaf", 1),
                max_features=params.get("max_features", "sqrt")))])
        if self.key == "adaboost":
            return Pipeline([imputer, ("clf", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=params.get("max_depth", 1)),
                n_estimators=params.get("n_estimators", 300), learning_rate=params.get("learning_rate", 0.1),
                random_state=random_state))])
        if self.key == "xgboost":
            return Pipeline([imputer, ("clf", XGBClassifier(
                n_estimators=params.get("n_estimators", 400), max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05), subsample=0.8, colsample_bytree=0.8,
                min_child_weight=params.get("min_child_weight", 1), reg_lambda=1.0,
                scale_pos_weight=neg / pos, eval_metric="aucpr", n_jobs=-1, random_state=random_state,
                tree_method="hist"))])
        raise KeyError(self.key)

    def fit(self, params: dict, X, y, random_state: int = 42) -> Pipeline:
        model = self.build(params, y, random_state)
        if self.key == "adaboost":
            model.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
        else:
            model.fit(X, y)
        return model


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("logistic_regression", "Logistic Regression", {"C": [0.05, 0.3, 1.0]}),
    ModelSpec("random_forest", "Random Forest", {"max_depth": [8, 16], "min_samples_leaf": [2, 5]}),
    ModelSpec("adaboost", "AdaBoost", {"max_depth": [1, 2], "learning_rate": [0.1, 0.5]}),
    ModelSpec("xgboost", "XGBoost", {"max_depth": [3, 5], "learning_rate": [0.03, 0.08], "min_child_weight": [1, 5]}),
]
SPEC_BY_KEY = {s.key: s for s in MODEL_SPECS}


def config_label(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "default"


__all__ = ["MODEL_SPECS", "SPEC_BY_KEY", "ModelSpec", "config_label", "clone", "signed_log1p"]
