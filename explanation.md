# IntelliFlow AutoML Engine: Code Explanation

This document explains the concepts, architecture, and code structure of the IntelliFlow AutoML Engine (Engine 1) located in `engines/automl/`.

## High-Level Architecture

The AutoML Engine is designed to take raw tabular data and a target column, and automatically produce the best possible machine learning model. It achieves this through a sequential pipeline:

1. **Preprocessing (`preprocessor.py`)**: Cleans the data, fills missing values, and encodes text/categorical data into numbers.
2. **Feature Engineering (`feature_eng.py`)**: Filters out useless features and optionally creates new mathematical features to help the model learn better.
3. **Hyperparameter Optimization (`hpo.py`)**: Uses a smart search algorithm (Optuna) to test thousands of different model settings to find the absolute best combination.
4. **Tracking & Registration (`tracker.py`)**: Logs every single test, metric, and model to a database (MLflow) so we can compare and deploy the winner.
5. **Pipeline Integration (`pipeline.py`)**: The master script that glues all the above steps together.

---

## Detailed Component Breakdown

### 1. `preprocessor.py`
**Core Concept:** Machine learning models only understand numbers. Data usually comes with missing values, dates, and text (categories). Preprocessing translates human data into machine-readable data.

- **`ColumnSchema` Data Class**: A simple container that categorizes columns into `numeric`, `categorical`, `datetime`, and `boolean`.
- **`infer_schema()`**: Automatically scans the DataFrame to figure out what type of data is in each column. It even uses regex to detect if a string column actually contains dates!
- **`_prepare_datetime_features()`**: Converts raw datetime columns (like `2023-10-01`) into separate numerical columns: Year (`2023`), Month (`10`), Day (`1`), etc.
- **`_build_transformer()`**: This is the heart of the file. It uses scikit-learn's `ColumnTransformer` to apply different rules to different columns simultaneously:
  - **Numeric columns**: Fills missing values using the median (`SimpleImputer`), and optionally scales them to have a mean of 0 and a standard deviation of 1 (`StandardScaler`).
  - **Low-cardinality categorical columns**: Uses `OneHotEncoder` to turn categories into binary (0/1) columns (e.g., Color_Red, Color_Blue).
  - **High-cardinality categorical columns**: Uses `OrdinalEncoder` to turn categories into integers (1, 2, 3...) to prevent creating thousands of columns.

### 2. `feature_eng.py`
**Core Concept:** Not all data is useful. Some columns are purely noise. Feature engineering strips away noise and highlights the important signals.

- **`VarianceThreshold`**: The first step in the pipeline. If a column has the exact same value for 99% of the rows (very low variance), it's useless for learning. This removes it.
- **`PolynomialFeatures` (Optional)**: Creates new columns by multiplying existing columns together. For example, if you have `Height` and `Width`, it might create `Height * Width` (Area).
- **`RFE` (Recursive Feature Elimination - Optional)**: Trains a quick Random Forest model, looks at which features it relied on the least, and deletes them. It repeats this until only the best features are left.
- **`PCA` (Principal Component Analysis - Optional)**: Compresses the data mathematically, reducing the number of columns while retaining the variance/information.

### 3. `tracker.py`
**Core Concept:** When running hundreds of experiments, you need a lab notebook to remember what settings resulted in what accuracy. MLflow is our digital lab notebook.

- **`MLflowTracker` Class**: A wrapper around the MLflow library. It's built with "lazy loading", meaning it only tries to import MLflow when it's actually running, preventing crashes if MLflow isn't installed.
- **`log_trial()`**: Called every time Optuna tests a new model. It logs the hyperparameters (e.g., `max_depth=5`), the resulting accuracy, and saves images of the Confusion Matrix and Feature Importances.
- **`log_best_model()`**: Once all trials are done, this function takes the absolute best model and explicitly saves the serialized `.pkl` model file to the MLflow Model Registry, making it ready for an API endpoint to serve.

### 4. `hpo.py`
**Core Concept:** We don't know if a Random Forest, XGBoost, or LightGBM model will work best. We also don't know what settings (hyperparameters) they should use. `hpo.py` uses Bayesian Optimization (Optuna) to intelligently guess and check.

- **`_objective()`**: The function Optuna tries to maximize. It picks a model family, guesses some hyperparameters, trains the model using Cross-Validation (K-Fold), and returns the average score.
- **`_suggest_params()`**: Defines the "Search Space". For example, it tells Optuna: "If you pick XGBoost, you can guess any `learning_rate` between 0.001 and 0.3".
- **`_build_model()`**: Takes the dictionary of guessed parameters and actually instantiates the scikit-learn/XGBoost/LightGBM python object.
- **`optimize()`**: The main loop. It runs `_objective()` `n_trials` times, keeps track of the best score, and then retrains the absolute best model on the entire training dataset.

### 5. `pipeline.py`
**Core Concept:** Orchestration.

- **`run_automl()`**: 
  1. Initializes tracking (`tracker.setup()`).
  2. Splits the target column from the feature columns (`split_features_target`).
  3. Preprocesses the data.
  4. Runs feature engineering.
  5. Passes the clean, engineered data to the `HPOEngine`.
  6. Returns a summary dictionary containing the `run_id`, `best_score`, and the `endpoint_url` placeholder.
