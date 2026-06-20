"""Model registry utilities for IntelliFlow AutoML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


DEFAULT_REGISTERED_MODEL_NAME = "intelliflow-automl-full-pipeline"


@dataclass
class RegistryRecord:
    """Metadata for a registered AutoML pipeline."""

    run_id: str
    model_name: str
    model_uri: str
    model_version: str | None = None
    artifact_path: str = "full_pipeline"
    alias: str | None = None


@dataclass
class ModelInfo:
    """Loadable registered model metadata."""

    model_name: str
    model_uri: str
    latest_version: str | None = None
    alias: str = "production"


class AutoMLRegistry:
    """Register and load complete AutoML pipelines from MLflow.

    The registry stores the full inference object:

    raw input row -> fitted preprocessor -> fitted feature engineer -> model

    This matters because prediction data must receive exactly the same
    transformations that were learned during training.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
        tracking_uri: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self._mlflow = None

    def build_full_pipeline(self, preprocessor: Any, feature_engineer: Any, model: Any) -> Pipeline:
        """Create a fitted sklearn pipeline for prediction-time reuse."""

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("feature_engineer", feature_engineer),
                ("model", model),
            ]
        )

    def register_pipeline(
        self,
        *,
        full_pipeline: Pipeline,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        artifact_path: str = "full_pipeline",
        alias: str = "production",
    ) -> RegistryRecord:
        """Log and register a complete fitted pipeline in MLflow."""

        mlflow = self._get_mlflow()
        with mlflow.start_run(run_name="registered-full-automl-pipeline") as run:
            for key, value in (tags or {}).items():
                mlflow.set_tag(key, str(value))

            mlflow.log_params(self._flatten_params(params))
            for key, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(key, float(value))

            model_info = mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                artifact_path=artifact_path,
                registered_model_name=self.model_name,
            )

            record = RegistryRecord(
                run_id=run.info.run_id,
                model_name=self.model_name,
                model_uri=model_info.model_uri,
                artifact_path=artifact_path,
                alias=alias,
            )

        record.model_version = self._find_model_version(record.run_id)
        if record.model_version:
            self.set_alias(record.model_version, alias)
        return record

    def load_model(self, model_uri: str | None = None):
        """Load a registered pipeline.

        If ``model_uri`` is omitted, the method tries the production alias first
        and then falls back to the latest registered version.
        """

        mlflow = self._get_mlflow()
        uri = model_uri or self.get_model_uri()
        return mlflow.sklearn.load_model(uri)

    def predict(self, rows: pd.DataFrame | list[dict[str, Any]], model_uri: str | None = None):
        """Load a registered pipeline and predict for raw input rows."""

        data = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        model = self.load_model(model_uri)
        return model.predict(data)

    def get_model_uri(self, alias: str = "production") -> str:
        """Return a loadable MLflow model URI for the registered pipeline."""

        client = self._get_client()
        try:
            client.get_model_version_by_alias(self.model_name, alias)
            return f"models:/{self.model_name}@{alias}"
        except Exception:
            version = self.get_latest_version()
            if version is None:
                raise RuntimeError(f"No registered versions found for {self.model_name!r}.")
            return f"models:/{self.model_name}/{version}"

    def get_model_info(self, alias: str = "production") -> ModelInfo:
        """Return model registry metadata for API responses."""

        return ModelInfo(
            model_name=self.model_name,
            model_uri=self.get_model_uri(alias=alias),
            latest_version=self.get_latest_version(),
            alias=alias,
        )

    def get_latest_version(self) -> str | None:
        """Find the newest registered model version."""

        client = self._get_client()
        versions = list(client.search_model_versions(f"name='{self.model_name}'"))
        if not versions:
            return None
        newest = max(versions, key=lambda item: int(getattr(item, "creation_timestamp", 0) or 0))
        return str(newest.version)

    def set_alias(self, version: str, alias: str = "production") -> None:
        """Point an MLflow alias such as production to a model version."""

        client = self._get_client()
        try:
            client.set_registered_model_alias(self.model_name, alias, version)
        except Exception:
            # Older MLflow installs may not support aliases. The model is still
            # registered and can be loaded by version.
            pass

    def _find_model_version(self, run_id: str) -> str | None:
        client = self._get_client()
        versions = list(client.search_model_versions(f"name='{self.model_name}'"))
        matching = [item for item in versions if item.run_id == run_id]
        if not matching:
            return None
        newest = max(matching, key=lambda item: int(getattr(item, "creation_timestamp", 0) or 0))
        return str(newest.version)

    def _get_mlflow(self):
        if self._mlflow is None:
            try:
                import mlflow
                import mlflow.sklearn
            except ImportError as exc:
                raise ImportError("MLflow is not installed. Run: pip install mlflow") from exc

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow = mlflow
        return self._mlflow

    def _get_client(self):
        self._get_mlflow()
        from mlflow.tracking import MlflowClient

        return MlflowClient()

    @staticmethod
    def _flatten_params(params: dict[str, Any]) -> dict[str, str | int | float | bool]:
        flat: dict[str, str | int | float | bool] = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                flat[key] = "None" if value is None else value
            else:
                flat[key] = str(value)
        return flat
