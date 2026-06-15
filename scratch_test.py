import pandas as pd
from sklearn.datasets import load_iris
from engines.automl.pipeline import run_automl

if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Run AutoML pipeline
    result = run_automl(dataset=df, target_column='target', metric='accuracy', n_trials=5)
    
    print("\n--- Pipeline Result ---")
    print(f"Run ID: {result['run_id']}")
    print(f"Best Score: {result['best_score']}")
    print(f"Endpoint URL: {result['endpoint_url']}")
