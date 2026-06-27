"""MLflow tracking helpers for IntelliFlow AutoML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


@dataclass
class MLflowRunRecord:
    """Small record returned after logging an MLflow run."""

    run_id: str | None
    artifact_paths: list[str] = field(default_factory=list)


class MLflowTracker:
    """Logs AutoML trials, metrics, artifacts, and best models to MLflow.

    The class imports MLflow lazily, so the rest of Engine 1 can still be
    imported on machines where MLflow is not installed yet. Install MLflow to
    enable real tracking:

    ``pip install mlflow``
    """

    def __init__(
        self,
        experiment_name: str = "intelliflow-automl",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled
        self._mlflow = None

    def setup(self) -> None:
        if not self.enabled:
            return

        try:
            import mlflow
            import mlflow.sklearn
        except ImportError as exc:
            raise ImportError("MLflow is not installed. Run: pip install mlflow") from exc

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._mlflow = mlflow

    def log_trial(
        self,
        *,
        trial_number: int,
        model_family: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        model: Any | None = None,
        feature_names: list[str] | None = None,
        y_true: pd.Series | np.ndarray | None = None,
        y_pred: pd.Series | np.ndarray | None = None,
        tags: dict[str, str] | None = None,
    ) -> MLflowRunRecord:
        """Log one HPO trial as a nested MLflow run."""

        if not self.enabled:
            return MLflowRunRecord(run_id=None)
        mlflow = self._get_mlflow()

        artifact_paths: list[str] = []
        with mlflow.start_run(run_name=f"trial-{trial_number}", nested=True) as run:
            mlflow.set_tag("trial_number", str(trial_number))
            mlflow.set_tag("model_family", model_family)
            for key, value in (tags or {}).items():
                mlflow.set_tag(key, value)

            mlflow.log_params(self._flatten_params(params))
            for key, value in metrics.items():
                if value is not None and np.isfinite(value):
                    mlflow.log_metric(key, float(value))

            if model is not None:
                mlflow.sklearn.log_model(model, artifact_path="model")
                artifact_paths.append("model")

            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                if y_true is not None and y_pred is not None:
                    cm_path = self._save_confusion_matrix(y_true, y_pred, tmp_path)
                    if cm_path:
                        mlflow.log_artifact(str(cm_path), artifact_path="plots")
                        artifact_paths.append("plots/confusion_matrix.png")

                importance_path = self._save_feature_importance(model, feature_names, tmp_path)
                if importance_path:
                    mlflow.log_artifact(str(importance_path), artifact_path="plots")
                    artifact_paths.append("plots/feature_importance.png")

            return MLflowRunRecord(run_id=run.info.run_id, artifact_paths=artifact_paths)

    def log_best_model(
        self,
        *,
        model: Any,
        model_family: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        registered_model_name: str = "intelliflow-best-model",
        tags: dict[str, str] | None = None,
    ) -> MLflowRunRecord:
        """Log and register the best AutoML model."""

        if not self.enabled:
            return MLflowRunRecord(run_id=None)
        mlflow = self._get_mlflow()

        with mlflow.start_run(run_name=f"best-{model_family}") as run:
            mlflow.set_tag("model_family", model_family)
            mlflow.set_tag("stage", "best")
            for key, value in (tags or {}).items():
                mlflow.set_tag(key, value)

            mlflow.log_params(self._flatten_params(params))
            for key, value in metrics.items():
                if value is not None and np.isfinite(value):
                    mlflow.log_metric(key, float(value))

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
            return MLflowRunRecord(run_id=run.info.run_id, artifact_paths=["model"])

    def _get_mlflow(self):
        if self._mlflow is None:
            self.setup()
        return self._mlflow

    @staticmethod
    def _flatten_params(params: dict[str, Any]) -> dict[str, str | int | float | bool]:
        flat: dict[str, str | int | float | bool] = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                flat[key] = "None" if value is None else value
            else:
                flat[key] = str(value)
        return flat

    @staticmethod
    def _save_confusion_matrix(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        output_dir: Path,
    ) -> Path | None:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        if labels.size > 20:
            return None

        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
        display.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        path = output_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    @staticmethod
    def _save_feature_importance(
        model: Any | None,
        feature_names: list[str] | None,
        output_dir: Path,
        top_n: int = 25,
    ) -> Path | None:
        if model is None or not hasattr(model, "feature_importances_"):
            return None

        importances = np.asarray(model.feature_importances_)
        names = feature_names or [f"feature_{i}" for i in range(len(importances))]
        if len(names) != len(importances):
            return None

        order = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(9, max(4, len(order) * 0.28)))
        plt.barh(np.asarray(names)[order], importances[order])
        plt.xlabel("Importance")
        plt.title("Top Feature Importances")
        plt.tight_layout()

        path = output_dir / "feature_importance.png"
        plt.savefig(path, dpi=160)
        plt.close()
        return path

