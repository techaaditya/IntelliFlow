import pandas as pd
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer:
    def __init__(self, variance_threshold=0.01, use_pca=False, n_components=None):
        self.variance_threshold = variance_threshold
        self.use_pca = use_pca
        self.n_components = n_components
        
        self.var_filter = VarianceThreshold(threshold=self.variance_threshold)
        self.pca = PCA(n_components=self.n_components) if self.use_pca else None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        X_transformed = X.copy()
        
        # Variance Filtering
        X_filtered = self.var_filter.fit_transform(X_transformed)
        kept_columns = X.columns[self.var_filter.get_support()]
        X_transformed = pd.DataFrame(X_filtered, columns=kept_columns, index=X.index)
        
        # Optional PCA
        if self.use_pca:
            X_pca = self.pca.fit_transform(X_transformed)
            pca_cols = [f'pca_{i}' for i in range(X_pca.shape[1])]
            X_transformed = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
            
        return X_transformed

    def transform(self, X: pd.DataFrame):
        X_transformed = X.copy()
        
        X_filtered = self.var_filter.transform(X_transformed)
        kept_columns = X.columns[self.var_filter.get_support()]
        X_transformed = pd.DataFrame(X_filtered, columns=kept_columns, index=X.index)
        
        if self.use_pca:
            X_pca = self.pca.transform(X_transformed)
            pca_cols = [f'pca_{i}' for i in range(X_pca.shape[1])]
            X_transformed = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
            
        return X_transformed
