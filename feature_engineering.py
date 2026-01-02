import pandas as pd
import numpy as np

INPUT_FILE = "training_data_flat.csv"
OUTPUT_FILE = "training_data_rolling.csv"
WINDOW_SIZE = 5


def main():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Sort by ESN and Cycle just in case
    df = df.sort_values(["ESN", "Cycles_Since_New"])

    # Identify sensor columns
    # We want to roll all 'Sensed_*' columns
    sensor_cols = [c for c in df.columns if c.startswith("Sensed_")]
    print(f"Found {len(sensor_cols)} sensor columns to roll.")

    # Group by ESN
    grouped = df.groupby("ESN")[sensor_cols]

    # Calculate Rolling Mean
    print(f"Calculating Rolling Mean (window={WINDOW_SIZE})...")
    rolling_mean = (
        grouped.rolling(window=WINDOW_SIZE, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_mean = rolling_mean.add_suffix(f"_mean_{WINDOW_SIZE}")

    # Calculate Rolling Std
    print(f"Calculating Rolling Std (window={WINDOW_SIZE})...")
    rolling_std = (
        grouped.rolling(window=WINDOW_SIZE, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    rolling_std = rolling_std.add_suffix(f"_std_{WINDOW_SIZE}")

    # Concatenate features
    print("Merging new features...")
    # Since we sorted and didn't drop rows, we can contact usually.
    # But safer to join by index if we preserved it.
    # Dataframe reset_index(drop=True) on rolling might act weird if original index wasn't range.
    # Let's verify alignement.

    df_out = pd.concat([df, rolling_mean, rolling_std], axis=1)

    # Handle NaNs (First N-1 rows might be NaN for Std if min_periods > 1, or just 1st row for std)
    # With min_periods=1, Std of 1 element is NaN.
    # We fill NaNs with 0 or backward fill?
    # Backward fill involves future variance ?? No, ffill?
    # For std of first point, it is undefined (NaN). Let's fill with 0.
    df_out = df_out.fillna(0)

    print(f"New shape: {df_out.shape}")

    print(f"Saving to {OUTPUT_FILE}...")
    df_out.to_csv(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
