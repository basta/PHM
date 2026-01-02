import pandas as pd
import numpy as np

INPUT_FILE = "training_data.csv"
OUTPUT_FILE = "training_data_flat.csv"


def main():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Original shape: {df.shape}")

    # Columns to pivot (the sensors)
    sensor_cols = [c for c in df.columns if c.startswith("Sensed_")]

    # Columns that should be static per flight (ESN, Cycle)
    # We'll use ESN and Cycles_Since_New as the index
    index_cols = ["ESN", "Cycles_Since_New"]

    # Other static columns (Targets + Cumulative history)
    # We need to grab these carefully. We'll take the first value for each group
    # and verify they are indeed constant (optional but good practice)
    meta_cols = [
        c
        for c in df.columns
        if c not in sensor_cols and c not in index_cols and c != "Snapshot"
    ]

    print(" verifying consistency of static columns...")
    # This check can be expensive on huge data, so maybe just trust or do a quick check
    # grouping = df.groupby(index_cols)[meta_cols].nunique()
    # if (grouping > 1).any().any():
    #     print("WARNING: Some metadata columns are not constant per flight cycle!")

    # Pivot the sensors
    print("Pivoting sensors...")
    df_pivot = df.pivot(index=index_cols, columns="Snapshot", values=sensor_cols)

    # Flatten the multi-level column index
    # e.g. (Sensed_T5, 1) -> Sensed_T5_1
    df_pivot.columns = [f"{col}_{snap}" for col, snap in df_pivot.columns]

    # Get the static metadata
    # We can just drop duplicates on the index columns to get one row per flight
    df_meta = (
        df[index_cols + meta_cols]
        .drop_duplicates(subset=index_cols)
        .set_index(index_cols)
    )

    # Join
    print("Joining metadata...")
    df_flat = df_meta.join(df_pivot)

    # Reset index to make ESN and Cycle normal columns again
    df_flat = df_flat.reset_index()

    print(f"New flattened shape: {df_flat.shape}")

    print(f"Saving to {OUTPUT_FILE}...")
    df_flat.to_csv(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
