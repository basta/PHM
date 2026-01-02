import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

INPUT_FILE = "training_data_flat.csv"
OUTPUT_FILE = "training_data_pca.csv"


class SensorReducer:
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

    def fit(self, df):
        self.feature_cols = self._get_feature_cols(df)
        X = df[self.feature_cols].values

        # Fit imputer and transform
        X = self.imputer.fit_transform(X)

        # Fit scaler and transform
        X = self.scaler.fit_transform(X)

        # Fit PCA
        self.pca.fit(X)

        self._is_fitted = True
        return self

    def transform(self, df):
        if not self._is_fitted:
            raise RuntimeError("SensorReducer has not been fitted yet.")

        # Ensure we use the same columns
        # If columns are missing, this will error, which is expected behavior
        X = df[self.feature_cols].values

        # Apply pipeline
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        X_pca = self.pca.transform(X)

        # Create output dataframe
        n_pcs = X_pca.shape[1]
        pc_cols = [f"PC_{i + 1}" for i in range(n_pcs)]
        return pd.DataFrame(X_pca, columns=pc_cols, index=df.index)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def main():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Initialize Reducer
    reducer = SensorReducer(n_components=0.95)

    print("Fitting and transforming data...")
    df_pca = reducer.fit_transform(df)

    # Metadata columns (everything that is NOT a sensor feature)
    meta_cols = [c for c in df.columns if c not in reducer.feature_cols]

    print(f"Original Features: {len(reducer.feature_cols)}")
    print(f"Reduced Components: {df_pca.shape[1]}")
    print(
        f"Explained Variance Ratio (sum): {np.sum(reducer.pca.explained_variance_ratio_):.4f}"
    )

    # Concatenate metadata and PCA features
    df_out = pd.concat([df[meta_cols], df_pca], axis=1)

    print(f"Saving reduced dataset to {OUTPUT_FILE}...")
    df_out.to_csv(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
