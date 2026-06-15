"""Feature engineering utilities for IntelliFlow AutoML."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

TaskType = Literal["classification", "regression"]


class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-style feature engineering block for processed tabular data.

    The expected input is the numeric DataFrame produced by
    ``AutoPreprocessor``. Operations run in this order:

    1. Variance filtering
    2. Optional polynomial feature expansion
    3. Optional recursive feature elimination
    4. Optional PCA dimensionality reduction
    """

    def __init__(
        self,
        variance_threshold: float = 0.0,
        polynomial_degree: int | None = None,
        polynomial_include_bias: bool = False,
        use_rfe: bool = False,
        rfe_n_features: int | float | None = None,
        rfe_estimator: BaseEstimator | None = None,
        task_type: TaskType = "classification",
        use_pca: bool = False,
        pca_n_components: int | float | None = None,
        random_state: int = 42,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.polynomial_degree = polynomial_degree
        self.polynomial_include_bias = polynomial_include_bias
        self.use_rfe = use_rfe
        self.rfe_n_features = rfe_n_features
        self.rfe_estimator = rfe_estimator
        self.task_type = task_type
        self.use_pca = use_pca
        self.pca_n_components = pca_n_components
        self.random_state = random_state

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> "AutoFeatureEngineer":
        frame = self._ensure_dataframe(X)
        self.input_features_ = list(frame.columns)

        steps: list[tuple[str, object]] = [
            ("variance", VarianceThreshold(threshold=self.variance_threshold)),
        ]

        if self.polynomial_degree and self.polynomial_degree > 1:
            steps.append(
                (
                    "polynomial",
                    PolynomialFeatures(
                        degree=self.polynomial_degree,
                        include_bias=self.polynomial_include_bias,
                    ),
                )
            )

        if self.use_rfe:
            if y is None:
                raise ValueError("RFE requires y during fit().")
            steps.append(("rfe", self._build_rfe(frame.shape[1])))

        if self.use_pca:
            steps.append(("pca", PCA(n_components=self.pca_n_components, random_state=self.random_state)))

        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(frame, y)
        self.feature_names_ = self._resolve_feature_names()
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        self._check_is_fitted()
        frame = self._ensure_dataframe(X)
        transformed = self.pipeline_.transform(frame)
        return pd.DataFrame(transformed, columns=self.feature_names_, index=frame.index)

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: object,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        self._check_is_fitted()
        return np.array(self.feature_names_, dtype=object)

    def selected_original_features(self) -> list[str]:
        """Return original features kept by the variance filter when available."""

        self._check_is_fitted()
        variance = self.pipeline_.named_steps["variance"]
        return list(np.array(self.input_features_, dtype=object)[variance.get_support()])

    def _build_rfe(self, original_feature_count: int) -> RFE:
        estimator = self.rfe_estimator
        if estimator is None:
            if self.task_type == "regression":
                estimator = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            else:
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)

        n_features = self.rfe_n_features
        if isinstance(n_features, float):
            if not 0 < n_features <= 1:
                raise ValueError("rfe_n_features as a float must be within (0, 1].")
            n_features = max(1, int(original_feature_count * n_features))
        elif n_features is None:
            n_features = max(1, original_feature_count // 2)

        return RFE(estimator=clone(estimator), n_features_to_select=n_features)

    def _resolve_feature_names(self) -> list[str]:
        names = np.array(self.input_features_, dtype=object)

        variance = self.pipeline_.named_steps["variance"]
        names = names[variance.get_support()]

        if "polynomial" in self.pipeline_.named_steps:
            poly = self.pipeline_.named_steps["polynomial"]
            names = poly.get_feature_names_out(names)

        if "rfe" in self.pipeline_.named_steps:
            rfe = self.pipeline_.named_steps["rfe"]
            names = names[rfe.get_support()]

        if "pca" in self.pipeline_.named_steps:
            pca = self.pipeline_.named_steps["pca"]
            names = np.array([f"pca_{i + 1}" for i in range(pca.n_components_)], dtype=object)

        return [str(name) for name in names]

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        raise TypeError("AutoFeatureEngineer expects a pandas DataFrame or numpy array.")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("AutoFeatureEngineer must be fitted before transform().")

