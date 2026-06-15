import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessor import AutoPreprocessor
from .feature_eng import FeatureEngineer
from .hpo import HPOEngine
from .tracker import log_trial_results
from .registry import register_best_model

def run_automl(dataset: pd.DataFrame, target_column: str, metric: str = 'accuracy', n_trials: int = 20):
    """
    Runs the full AutoML pipeline: preprocessing, feature engineering, HPO, and model registration.
    """
    print("Starting AutoML Pipeline...")
    
    # 1. Preprocessing
    print("Preprocessing data...")
    preprocessor = AutoPreprocessor(target_column=target_column)
    X, y = preprocessor.fit_transform(dataset)
    
    # 2. Feature Engineering
    print("Engineering features...")
    fe = FeatureEngineer(variance_threshold=0.01, use_pca=False)
    X_fe = fe.fit_transform(X)
    
    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_fe, y, test_size=0.2, random_state=42, stratify=y)
    
    # Start parent MLflow run
    with mlflow.start_run(run_name="AutoML_Pipeline") as parent_run:
        # 3. Hyperparameter Optimization
        print("Starting Hyperparameter Optimization...")
        hpo = HPOEngine(n_trials=n_trials, metric=metric)
        best_model, best_score = hpo.optimize(X_train, y_train)
        
        print(f"Best score ({metric}): {best_score}")
        print(f"Best model parameters: {hpo.best_params}")
        
        # 4. Final Evaluation and Tracking
        print("Evaluating best model...")
        best_model.fit(X_train, y_train)
        
        log_trial_results(
            params=hpo.best_params,
            metrics={'test_score': best_model.score(X_test, y_test)},
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            feature_names=X_train.columns.tolist()
        )
        
        # Log the final model
        mlflow.sklearn.log_model(best_model, "model")
        
        # 5. Model Registry
        print("Registering model...")
        try:
            version = register_best_model(parent_run.info.run_id)
            print(f"Model registered successfully. Version: {version}")
        except Exception as e:
            print(f"Could not register model: {e}")
            
    return {
        'run_id': parent_run.info.run_id,
        'best_score': best_score,
        'model': best_model,
        'endpoint_url': f"/automl/predict/{parent_run.info.run_id}" # Placeholder for actual endpoint
    }
