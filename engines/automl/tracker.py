import mlflow
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def log_trial_results(params, metrics, model=None, X_test=None, y_test=None, feature_names=None):
    """
    Logs parameters, metrics, and artifacts to the current MLflow run.
    """
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    # Optionally log the model and artifacts if provided
    if model is not None and X_test is not None and y_test is not None:
        # Generate and log confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        if os.path.exists(cm_path):
            os.remove(cm_path)
            
        # Log feature importance if supported by model
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
            
            plt.figure(figsize=(10, 6))
            feat_imp.plot(kind='bar')
            plt.title('Feature Importances')
            plt.tight_layout()
            fi_path = "feature_importance.png"
            plt.savefig(fi_path)
            plt.close()
            mlflow.log_artifact(fi_path)
            if os.path.exists(fi_path):
                os.remove(fi_path)
