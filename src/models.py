import xgboost as xgb
from sklearn.base import BaseEstimator


def get_model(model_name: str, random_state: int = 42, **kwargs) -> BaseEstimator:
    if model_name == "xgboost":
        # Default parameters from baseline
        params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "n_jobs": -1,
            "verbosity": 0,
            "random_state": random_state,
        }
        # Update with kwargs
        params.update(kwargs)

        return xgb.XGBRegressor(**params)

    raise ValueError(f"Unknown model architecture: {model_name}")
