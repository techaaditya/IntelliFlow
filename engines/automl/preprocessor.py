import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

class AutoPreprocessor:
    def __init__(self, target_column: str = None):
        self.target_column = target_column
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []

    def fit_transform(self, df: pd.DataFrame):
        X = df.copy()
        y = None
        
        if self.target_column and self.target_column in X.columns:
            y = X.pop(self.target_column)

        # Detect types
        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # Using OrdinalEncoder for simplicity and tree-based models
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names if possible
        feature_names = self.numeric_features + self.categorical_features
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

        return X_processed_df, y

    def transform(self, df: pd.DataFrame):
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        
        X = df.copy()
        if self.target_column and self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])
            
        X_processed = self.preprocessor.transform(X)
        feature_names = self.numeric_features + self.categorical_features
        return pd.DataFrame(X_processed, columns=feature_names, index=X.index)
