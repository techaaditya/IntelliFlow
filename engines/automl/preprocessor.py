"""Automatic preprocessing for IntelliFlow AutoML.

This module prepares raw tabular data for model training. It detects column
types, imputes missing values, encodes categorical features, and scales numeric
features using sklearn-compatible transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

EncodingStrategy = Literal["onehot", "ordinal"]
ScalingStrategy = Literal["standard", "none"]


@dataclass
class ColumnSchema:
    """Detected feature schema for a dataset."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """Detects feature types and builds an sklearn preprocessing pipeline.

    Parameters
    ----------
    encoding:
        ``"onehot"`` for nominal categorical features or ``"ordinal"`` for
        tree models where compact encoded columns are preferred.
    scaling:
        ``"standard"`` applies ``StandardScaler`` to numeric columns.
    datetime_features:
        When true, datetime columns are expanded into year, month, day,
        day-of-week, and hour numeric features.
    max_onehot_cardinality:
        Columns above this cardinality are ordinal-encoded even when
        ``encoding="onehot"`` to prevent very wide matrices.
    """

    def __init__(
        self,
        encoding: EncodingStrategy = "onehot",
        scaling: ScalingStrategy = "standard",
        datetime_features: bool = True,
        max_onehot_cardinality: int = 50,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
    ) -> None:
        self.encoding = encoding
        self.scaling = scaling
        self.datetime_features = datetime_features
        self.max_onehot_cardinality = max_onehot_cardinality
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> "AutoPreprocessor":
        frame = self._normalize_missing(self._ensure_dataframe(X))
        self.schema_ = self.infer_schema(frame)
        prepared = self._prepare_datetime_features(frame, fit=True)
        self.transformer_ = self._build_transformer(prepared)
        self.transformer_.fit(prepared, y)
        self.feature_names_ = self._resolve_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        frame = self._normalize_missing(self._ensure_dataframe(X))
        prepared = self._prepare_datetime_features(frame, fit=False)
        transformed = self.transformer_.transform(prepared)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        return pd.DataFrame(transformed, columns=self.feature_names_, index=frame.index)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: object,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def infer_schema(self, X: pd.DataFrame) -> ColumnSchema:
        schema = ColumnSchema()
        for col in X.columns:
            series = X[col]
            if pd.api.types.is_bool_dtype(series):
                schema.boolean.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                schema.numeric.append(col)
            elif pd.api.types.is_datetime64_any_dtype(series):
                schema.datetime.append(col)
            elif self._looks_like_datetime(series):
                schema.datetime.append(col)
            else:
                schema.categorical.append(col)
        return schema

    def get_feature_names_out(self) -> np.ndarray:
        self._check_is_fitted()
        return np.array(self.feature_names_, dtype=object)

    def _build_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = [c for c in X.columns if c.startswith("__dt_") or c in self.schema_.numeric]
        boolean_cols = [c for c in self.schema_.boolean if c in X.columns]
        categorical_cols = [c for c in self.schema_.categorical if c in X.columns]

        low_cardinality, high_cardinality = self._split_by_cardinality(X, categorical_cols)

        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy=self.numeric_impute_strategy)),
        ]
        if self.scaling == "standard":
            numeric_steps.append(("scaler", StandardScaler()))

        categorical_onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )

        transformers: list[tuple[str, object, list[str]]] = []
        if numeric_cols:
            transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
        if boolean_cols:
            transformers.append(
                (
                    "boolean",
                    Pipeline(
                        [
                            (
                                "as_object",
                                FunctionTransformer(
                                    lambda data: data.astype(object),
                                    feature_names_out="one-to-one",
                                ),
                            ),
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ordinal", OrdinalEncoder()),
                        ]
                    ),
                    boolean_cols,
                )
            )
        if self.encoding == "onehot" and low_cardinality:
            transformers.append(
                (
                    "categorical_onehot",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy=self.categorical_impute_strategy)),
                            ("onehot", categorical_onehot),
                        ]
                    ),
                    low_cardinality,
                )
            )
        ordinal_cols = high_cardinality if self.encoding == "onehot" else categorical_cols
        if ordinal_cols:
            transformers.append(
                (
                    "categorical_ordinal",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy=self.categorical_impute_strategy)),
                            ("ordinal", categorical_ordinal),
                        ]
                    ),
                    ordinal_cols,
                )
            )

        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    def _prepare_datetime_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        frame = X.copy()
        datetime_cols = getattr(self, "schema_", self.infer_schema(frame)).datetime

        if fit:
            self.datetime_columns_ = datetime_cols if self.datetime_features else []

        for col in getattr(self, "datetime_columns_", []):
            if col not in frame.columns:
                continue
            dt = pd.to_datetime(frame[col], errors="coerce")
            prefix = f"__dt_{col}"
            frame[f"{prefix}_year"] = dt.dt.year
            frame[f"{prefix}_month"] = dt.dt.month
            frame[f"{prefix}_day"] = dt.dt.day
            frame[f"{prefix}_dayofweek"] = dt.dt.dayofweek
            frame[f"{prefix}_hour"] = dt.dt.hour
            frame = frame.drop(columns=[col])

        return frame

    def _split_by_cardinality(
        self,
        X: pd.DataFrame,
        columns: Iterable[str],
    ) -> tuple[list[str], list[str]]:
        low_cardinality: list[str] = []
        high_cardinality: list[str] = []
        for col in columns:
            unique_count = X[col].nunique(dropna=True)
            if unique_count <= self.max_onehot_cardinality:
                low_cardinality.append(col)
            else:
                high_cardinality.append(col)
        return low_cardinality, high_cardinality

    def _resolve_feature_names(self) -> list[str]:
        names = self.transformer_.get_feature_names_out()
        return [str(name) for name in names]

    @staticmethod
    def _looks_like_datetime(series: pd.Series) -> bool:
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            return False
        sample = series.dropna().astype(str).head(25)
        if sample.empty:
            return False
        date_like = sample.str.contains(r"\d{4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True)
        if date_like.mean() < 0.8:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() >= 0.8

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AutoPreprocessor expects a pandas DataFrame.")
        return X

    @staticmethod
    def _normalize_missing(X: pd.DataFrame) -> pd.DataFrame:
        return X.replace({None: np.nan})

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "transformer_"):
            raise RuntimeError("AutoPreprocessor must be fitted before transform().")


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``X, y`` and validate that the target column exists."""

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} was not found in the dataset.")
    return df.drop(columns=[target_col]), df[target_col]
