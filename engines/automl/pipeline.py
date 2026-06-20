import pandas as pd

from .preprocessor import AutoPreprocessor, split_features_target
from .feature_eng import AutoFeatureEngineer
from .hpo import HPOEngine
from .registry import AutoMLRegistry, DEFAULT_REGISTERED_MODEL_NAME
from .tracker import MLflowTracker

def run_automl(
    dataset: pd.DataFrame,
    target_column: str,
    metric: str = 'accuracy',
    n_trials: int = 20,
    task_type: str = "classification",
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
):
    """
    Runs the full AutoML pipeline: preprocessing, feature engineering, HPO, and model registration.
    """
    print("Starting AutoML Pipeline...")
    if task_type == "auto":
        task_type = detect_task_type(dataset[target_column])
    metric = metric or ("accuracy" if task_type == "classification" else "r2")
    
    # 1. Setup MLflow Tracking
    tracker = MLflowTracker(experiment_name="intelliflow-automl", enabled=True)
    tracker.setup()
    
    # 2. Extract features and target
    X, y = split_features_target(dataset, target_column)
    
    # 3. Preprocessing
    print("Preprocessing data...")
    preprocessor = AutoPreprocessor(encoding="onehot", scaling="standard")
    X_processed = preprocessor.fit_transform(X, y)
    
    # 4. Feature Engineering
    print("Engineering features...")
    fe = AutoFeatureEngineer(variance_threshold=0.01, use_pca=False, task_type=task_type)
    X_fe = fe.fit_transform(X_processed, y)
    
    # 5. Hyperparameter Optimization & Model Registration
    # The HPOEngine automatically handles evaluation and calls tracker.log_best_model() to register the model
    print("Starting Hyperparameter Optimization...")
    hpo = HPOEngine(
        task_type=task_type,
        metric=metric,
        n_trials=n_trials,
        tracker=tracker
    )
    result = hpo.optimize(X_fe, y)
    
    print(f"Best score ({metric}): {result.best_score}")
    print(f"Best model family: {result.best_model_family}")

    # 6. Register the complete prediction pipeline, not just the final model.
    # New raw rows must go through the same fitted preprocessing and feature
    # engineering steps before the model can predict.
    print("Registering full AutoML pipeline...")
    registry = AutoMLRegistry(model_name=registered_model_name)
    full_pipeline = registry.build_full_pipeline(preprocessor, fe, result.best_model)
    registry_record = registry.register_pipeline(
        full_pipeline=full_pipeline,
        params={
            "target_column": target_column,
            "metric": metric,
            "task_type": task_type,
            **result.best_params,
        },
        metrics={"best_score": result.best_score},
        tags={
            "engine": "automl",
            "model_family": result.best_model_family,
        },
    )
    run_id = registry_record.run_id
    
    return {
        'run_id': run_id,
        'best_score': result.best_score,
        'best_model_family': result.best_model_family,
        'best_params': result.best_params,
        'model': result.best_model,
        'full_pipeline': full_pipeline,
        'model_name': registry_record.model_name,
        'model_version': registry_record.model_version,
        'model_uri': registry_record.model_uri,
        'endpoint_url': f"/automl/predict/{registry_record.model_name}",
        'task_type': task_type,
        'metric': metric,
        'target_column': target_column,
        'n_rows': int(dataset.shape[0]),
        'n_columns': int(dataset.shape[1]),
    }


def predict_with_registered_model(rows: pd.DataFrame | list[dict], model_uri: str | None = None):
    """Predict with the latest registered full AutoML pipeline."""

    registry = AutoMLRegistry()
    return registry.predict(rows, model_uri=model_uri)


def detect_task_type(target: pd.Series) -> str:
    """Infer whether a supervised target is classification or regression."""

    clean = target.dropna()
    if clean.empty:
        raise ValueError("Cannot infer task type from an empty target column.")

    if not pd.api.types.is_numeric_dtype(clean):
        return "classification"

    unique_count = clean.nunique()
    unique_ratio = unique_count / len(clean)
    if unique_count <= 20 or unique_ratio <= 0.05:
        return "classification"
    return "regression"
