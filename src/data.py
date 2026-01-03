import pandas as pd
import numpy as np

# Configuration
CSV_FILENAME = "training_data_flat.csv"

TARGET_COLUMNS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]

METADATA_COLUMNS = (
    [
        "ESN",
        "Snapshot",
        "Cycles_Since_New",  # Usually an index or feature, but often treated as metadata or handled separately
        "Cumulative_WWs",
        "Cumulative_HPC_SVs",
        "Cumulative_HPT_SVs",
    ]
    + TARGET_COLUMNS
)


def load_data(path=CSV_FILENAME):
    print(f"Loading data from {path}...")
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset: {len(df):,} rows.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{path}'")
        raise


def get_feature_columns(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = METADATA_COLUMNS
    return [c for c in df.columns if c not in exclude_cols]


def get_logo_cv_folds(df):
    """
    Yields (train_mask, test_mask, test_esn)
    """
    unique_esns = df["ESN"].unique()
    for test_esn in unique_esns:
        train_mask = df["ESN"] != test_esn
        test_mask = df["ESN"] == test_esn
        yield train_mask, test_mask, test_esn
