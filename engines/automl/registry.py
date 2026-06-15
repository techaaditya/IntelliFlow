import mlflow
from mlflow.tracking import MlflowClient

def register_best_model(run_id, model_name="IntelliFlow_AutoML_Model"):
    """
    Registers the model from the best run to the MLflow Model Registry.
    """
    # Register the model
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        model_name
    )
    
    # Transition the model version to "Staging"
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )
    
    return result.version
