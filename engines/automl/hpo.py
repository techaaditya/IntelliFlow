import optuna
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .tracker import log_trial_results

class HPOEngine:
    def __init__(self, n_trials=100, metric='accuracy', n_splits=5):
        self.n_trials = n_trials
        self.metric = metric
        self.n_splits = n_splits
        self.best_model = None
        self.best_params = None

    def _objective(self, trial, X, y):
        model_name = trial.suggest_categorical('model', ['random_forest', 'xgboost', 'lightgbm'])
        
        if model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                max_depth=trial.suggest_int('max_depth', 3, 30, log=False) if trial.suggest_categorical('use_max_depth', [True, False]) else None,
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10)
            )
        elif model_name == 'xgboost':
            model = XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                max_depth=trial.suggest_int('max_depth', 3, 12),
                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0)
            )
        elif model_name == 'lightgbm':
            model = LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                num_leaves=trial.suggest_int('num_leaves', 20, 300),
                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
                feature_fraction=trial.suggest_float('feature_fraction', 0.5, 1.0),
                verbose=-1
            )

        with mlflow.start_run(nested=True):
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            
            scoring = self.metric
            if self.metric == 'roc_auc' and len(set(y)) > 2:
                scoring = 'roc_auc_ovr'
                
            score = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
            
            # Use tracker
            log_trial_results(
                params=trial.params,
                metrics={f'cv_{self.metric}': score}
            )
            
        return score

    def optimize(self, X, y):
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(lambda t: self._objective(t, X, y), n_trials=self.n_trials)
        
        self.best_params = study.best_params
        model_name = self.best_params.pop('model')
        
        # Reconstruct the best model
        if model_name == 'random_forest':
            self.best_params.pop('use_max_depth', None) # Clean up unused params
            self.best_model = RandomForestClassifier(**self.best_params)
        elif model_name == 'xgboost':
            self.best_model = XGBClassifier(**self.best_params)
        elif model_name == 'lightgbm':
            self.best_model = LGBMClassifier(**self.best_params)
            
        return self.best_model, study.best_value
