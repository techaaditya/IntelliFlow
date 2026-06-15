import pandas as pd
import mlflow

from .preprocessor import AutoPreprocessor, split_features_target
from .feature_eng import AutoFeatureEngineer
from .hpo import HPOEngine
from .tracker import MLflowTracker

def run_automl(dataset: pd.DataFrame, target_column: str, metric: str = 'accuracy', n_trials: int = 20, task_type: str = "classification"):
    """
    Runs the full AutoML pipeline: preprocessing, feature engineering, HPO, and model registration.
    """
    print("Starting AutoML Pipeline...")
    
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
    
    # Optionally get the latest run ID from MLflow to return to the user
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else "logged-in-hpo-engine"
    
    return {
        'run_id': run_id,
        'best_score': result.best_score,
        'model': result.best_model,
        'endpoint_url': f"/automl/predict/{run_id}" # Placeholder for API
    }
