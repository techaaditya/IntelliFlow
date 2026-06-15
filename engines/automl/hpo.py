"""Optuna hyperparameter optimization for IntelliFlow AutoML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score
from sklearn.metrics import r2_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from .tracker import MLflowTracker

TaskType = Literal["classification", "regression"]


@dataclass
class TrialResult:
    """Result for a single HPO trial."""

    trial_number: int
    model_family: str
    params: dict[str, Any]
    cv_score: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class HPOResult:
    """Final result returned by the HPO engine."""

    best_model: BaseEstimator
    best_model_family: str
    best_params: dict[str, Any]
    best_score: float
    metric: str
    task_type: TaskType
    trials: list[TrialResult]
    feature_names: list[str]


class HPOEngine:
    """Search Random Forest, XGBoost, and LightGBM with Optuna.

    XGBoost and LightGBM are optional. If their packages are not installed, the
    engine skips those model families and still searches Random Forest.
    """

    def __init__(
        self,
        task_type: TaskType = "classification",
        metric: str | None = None,
        n_trials: int = 50,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        model_families: list[str] | None = None,
        tracker: MLflowTracker | None = None,
    ) -> None:
        self.task_type = task_type
        self.metric = metric or ("roc_auc" if task_type == "classification" else "r2")
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.model_families = model_families or ["random_forest", "xgboost", "lightgbm"]
        self.tracker = tracker

    def optimize(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> HPOResult:
        """Run Optuna optimization and fit the best model."""

        optuna = self._load_optuna()
        frame = self._ensure_dataframe(X)
        target = pd.Series(y).reset_index(drop=True)
        frame = frame.reset_index(drop=True)
        if self.task_type == "classification" and self.metric == "roc_auc" and target.nunique() > 2:
            self.metric = "roc_auc_ovr_weighted"
        available_families = self.available_model_families()
        if not available_families:
            raise RuntimeError("No supported model families are available.")

        X_train, X_test, y_train, y_test = train_test_split(
            frame,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target if self.task_type == "classification" and target.nunique() > 1 else None,
        )

        self.trials_: list[TrialResult] = []

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.MedianPruner()
        direction = "maximize" if self._higher_is_better() else "minimize"
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: self._objective(trial, X_train, y_train, available_families), n_trials=self.n_trials)

        best_model_family = study.best_trial.params["model_family"]
        best_model_params = dict(study.best_trial.user_attrs["model_params"])
        best_model = self._build_model({"model_family": best_model_family, **best_model_params})
        best_model.fit(X_train, y_train)
        test_metrics = self.evaluate(best_model, X_test, y_test)

        if self.tracker:
            self.tracker.log_best_model(
                model=best_model,
                model_family=best_model_family,
                params={"model_family": best_model_family, **best_model_params},
                metrics={"best_cv_score": float(study.best_value), **test_metrics},
            )

        self.study_ = study
        self.best_model_ = best_model
        return HPOResult(
            best_model=best_model,
            best_model_family=best_model_family,
            best_params={"model_family": best_model_family, **best_model_params},
            best_score=float(study.best_value),
            metric=self.metric,
            task_type=self.task_type,
            trials=self.trials_,
            feature_names=list(frame.columns),
        )

    def available_model_families(self) -> list[str]:
        """Return requested model families that can run in this environment."""

        available = []
        requested = set(self.model_families)
        if "random_forest" in requested:
            available.append("random_forest")
        if "xgboost" in requested and self._module_exists("xgboost"):
            available.append("xgboost")
        if "lightgbm" in requested and self._module_exists("lightgbm"):
            available.append("lightgbm")
        return available

    def evaluate(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Compute test metrics for the fitted best model."""

        y_pred = model.predict(X_test)

        if self.task_type == "regression":
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            return {
                "test_rmse": rmse,
                "test_mae": float(mean_absolute_error(y_test, y_pred)),
                "test_r2": float(r2_score(y_test, y_pred)),
            }

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        if hasattr(model, "predict_proba") and pd.Series(y_test).nunique() == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_proba))

        return metrics

    def _objective(
        self,
        trial: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        available_families: list[str],
    ) -> float:
        model_family = trial.suggest_categorical("model_family", available_families)
        params = self._suggest_params(trial, model_family)
        trial.set_user_attr("model_params", params)
        model = self._build_model({"model_family": model_family, **params})
        cv = self._build_cv(y_train)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=self.metric, n_jobs=-1)
        score = float(np.mean(scores))

        result = TrialResult(
            trial_number=trial.number,
            model_family=model_family,
            params={"model_family": model_family, **params},
            cv_score=score,
            metrics={"cv_mean": score, "cv_std": float(np.std(scores))},
        )
        self.trials_.append(result)

        if self.tracker:
            trial_model = clone(model)
            trial_model.fit(X_train, y_train)
            self.tracker.log_trial(
                trial_number=trial.number,
                model_family=model_family,
                params=result.params,
                metrics=result.metrics,
                model=trial_model,
                feature_names=list(X_train.columns),
                tags={"metric": self.metric},
            )

        return score

    def _suggest_params(self, trial: Any, model_family: str) -> dict[str, Any]:
        if model_family == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("rf_n_estimators", 50, 500),
                "max_depth": trial.suggest_int("rf_max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
            }
            if trial.suggest_categorical("rf_use_max_depth_none", [False, True]):
                params["max_depth"] = None
            return params

        if model_family == "xgboost":
            return {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 500),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
                "learning_rate": trial.suggest_float("xgb_learning_rate", 0.001, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
            }

        if model_family == "lightgbm":
            return {
                "num_leaves": trial.suggest_int("lgbm_num_leaves", 20, 300),
                "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.001, 0.3, log=True),
                "n_estimators": trial.suggest_int("lgbm_n_estimators", 50, 500),
                "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 5, 100),
                "feature_fraction": trial.suggest_float("lgbm_feature_fraction", 0.5, 1.0),
            }

        raise ValueError(f"Unsupported model family: {model_family}")

    def _build_model(self, params: dict[str, Any]) -> BaseEstimator:
        model_family = params["model_family"]
        model_params = {key: value for key, value in params.items() if key != "model_family"}

        if model_family == "random_forest":
            cls = RandomForestClassifier if self.task_type == "classification" else RandomForestRegressor
            return cls(**model_params, random_state=self.random_state, n_jobs=-1)

        if model_family == "xgboost":
            if self.task_type == "classification":
                from xgboost import XGBClassifier

                return XGBClassifier(**model_params, random_state=self.random_state, eval_metric="logloss", n_jobs=-1)
            from xgboost import XGBRegressor

            return XGBRegressor(**model_params, random_state=self.random_state, n_jobs=-1)

        if model_family == "lightgbm":
            if self.task_type == "classification":
                from lightgbm import LGBMClassifier

                return LGBMClassifier(**model_params, random_state=self.random_state, n_jobs=-1, verbose=-1)
            from lightgbm import LGBMRegressor

            return LGBMRegressor(**model_params, random_state=self.random_state, n_jobs=-1, verbose=-1)

        raise ValueError(f"Unsupported model family: {model_family}")

    def _build_cv(self, y: pd.Series):
        if self.task_type == "classification":
            min_class_count = int(y.value_counts().min())
            folds = max(2, min(self.cv_folds, min_class_count))
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    def _higher_is_better(self) -> bool:
        return True

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        raise TypeError("HPOEngine expects a pandas DataFrame or numpy array.")

    @staticmethod
    def _module_exists(module_name: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None

    @staticmethod
    def _load_optuna():
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("Optuna is not installed. Run: pip install optuna") from exc
        return optuna
