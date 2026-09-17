"""Feature engineering for the penguins classifier.

The transformations live in a scikit-learn ColumnTransformer so that exactly the
same preprocessing is applied at training time and inside the API.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "species"
NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
CATEGORICAL_FEATURES = ["island", "sex"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]

# Engineered features added on top of the raw columns.
DERIVED_FEATURES = ["bill_ratio", "mass_per_flipper"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cheap ratio features that separate the species better than raw sizes."""
    out = df.copy()
    out["bill_ratio"] = out["bill_length_mm"] / out["bill_depth_mm"]
    out["mass_per_flipper"] = out["body_mass_g"] / out["flipper_length_mm"]
    return out


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Stateless transformer wrapping :func:`add_derived_features`.

    A class rather than a ``FunctionTransformer``: the fitted pipeline is
    pickled in the training container and unpickled in the API container, and a
    class pickles as a plain reference to this module, while a function attribute
    drags in whatever serialiser (dill/cloudpickle) happened to be installed
    where the model was trained.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "DerivedFeatures":  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return add_derived_features(X)


def build_preprocessor() -> ColumnTransformer:
    numeric = [*NUMERIC_FEATURES, *DERIVED_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    """Full inference pipeline: derived features -> preprocessing -> estimator."""
    return Pipeline(
        steps=[
            ("derived", DerivedFeatures()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
