import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class SensorReducer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.95, prefix="Sensed_"):
        self.n_components = n_components
        self.prefix = prefix
        self.feature_cols = None
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._is_fitted = False

    def _get_feature_cols(self, df):
        return [c for c in df.columns if c.startswith(self.prefix)]

    def fit(self, X, y=None):
        # X is expected to be a DataFrame
        self.feature_cols = self._get_feature_cols(X)
        X_data = X[self.feature_cols].values

        # Fit imputer and transform
        X_data = self.imputer.fit_transform(X_data)

        # Fit scaler and transform
        X_data = self.scaler.fit_transform(X_data)

        # Fit PCA
        self.pca.fit(X_data)

        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("SensorReducer has not been fitted yet.")

        # X is expected to be a DataFrame
        # Ensure we use the same columns
        X_data = X[self.feature_cols].values

        # Apply pipeline
        X_data = self.imputer.transform(X_data)
        X_data = self.scaler.transform(X_data)
        X_pca = self.pca.transform(X_data)

        # Create output dataframe
        n_pcs = X_pca.shape[1]
        pc_cols = [f"PC_{i + 1}" for i in range(n_pcs)]
        return pd.DataFrame(X_pca, columns=pc_cols, index=X.index)


class RollingFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size=5,
        prefix="Sensed_",
        group_col="ESN",
        time_col="Cycles_Since_New",
    ):
        self.window_size = window_size
        self.prefix = prefix
        self.group_col = group_col
        self.time_col = time_col
        self.sensor_cols = None
        self._is_fitted = False

    def _get_sensor_cols(self, df):
        return [c for c in df.columns if c.startswith(self.prefix)]

    def fit(self, X, y=None):
        # We identify the sensor columns to roll.
        # X must contain the group_col (ESN) and time_col (Cycles) to sort correctly,
        # plus the sensor columns.
        if self.group_col not in X.columns or self.time_col not in X.columns:
            raise ValueError(
                f"X must contain '{self.group_col}' and '{self.time_col}' columns for rolling features."
            )

        self.sensor_cols = self._get_sensor_cols(X)
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("RollingFeaturesTransformer has not been fitted.")

        df = X.copy()

        df = df.sort_values([self.group_col, self.time_col])

        grouped = df.groupby(self.group_col)[self.sensor_cols]

        rolling_mean = (
            grouped.rolling(window=self.window_size, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        ).add_suffix(f"_mean_{self.window_size}")

        rolling_std = (
            grouped.rolling(window=self.window_size, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        ).add_suffix(f"_std_{self.window_size}")

        rolling_std = rolling_std.fillna(0)


        new_features = pd.concat([rolling_mean, rolling_std], axis=1)

        X_out = X.join(new_features)
        return X_out
