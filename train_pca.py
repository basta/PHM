import pandas as pd
import numpy as np
import xgboost as xgb
import time
from dimensionality_reduction import SensorReducer

# --- 1. Configuration ---
CSV_FILENAME = "training_data_flat.csv"  
RANDOM_STATE = 42

# Define your three target columns
TARGET_COLUMNS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]

# Define your feature columns (all sensors, settings, and history)
METADATA_COLUMNS = [
    "ESN",
    "Snapshot",
    "Cumulative_WWs",
    "Cumulative_HPC_SVs",
    "Cumulative_HPT_SVs",
] + TARGET_COLUMNS


# --- 2. Scoring Function ---
def time_weighted_error(y_true, y_pred, alpha=0.02, beta=1):
    """
    Returns the weighted squared error.
    Late predictions (error >= 0) are penalized 2x more than early predictions.
    Weights increase as y_true (remaining cycles) decreases.
    """
    error = y_pred - y_true

    # Weight calculation based on proximity to event (y_true)
    # and direction of error (late vs early)
    weight = np.where(error >= 0, 2 / (1 + alpha * y_true), 1 / (1 + alpha * y_true))
    return weight * (error**2) * beta


def main():
    # --- 3. Load and Prepare Data ---
    print(f"Loading data from {CSV_FILENAME}...")
    try:
        df = pd.read_csv(CSV_FILENAME)
        print(f"Loaded full dataset: {len(df):,} rows.")
    except FileNotFoundError:
        print(f"Error: File not found at '{CSV_FILENAME}'")
        return

    # Drop rows with any missing values (though SensorReducer handles NaNs, targets might be missing?)
    # Assuming targets are always present based on problem_info.
    # We will let SensorReducer handle NaNs in X.

    # Define features dynamically - INITIALLY load all potential features
    # SensorReducer filters for "Sensed_" prefix internally.
    # However, we might want to keep other non-sensor features?
    # For this specific request "train on these reduced features",
    # and the PCA script only reduced "Sensed_" columns.
    # We should train on (Reduced Sensors) + (Metadata like Settings/History??).
    # The prompt said "train on *these* reduced features".
    # Let's stick to replacing Sensed_* with PCs, and keeping other valid features.

    # Identify non-sensor feature columns that we might want to keep (e.g. settings?)
    # In problem_info, Sensed_ includes settings like Sensed_VAFN.
    # So basically everything is "Sensed_".
    # There are no other features except metadata/targets.
    # So we train ONLY on PCs.

    y = df[TARGET_COLUMNS]

    # LOGO-CV
    unique_esns = df["ESN"].unique()
    print(f"Starting LOGO-CV with {len(unique_esns)} engines using PCA features.")

    cv_scores = {"WW": [], "HPC": [], "HPT": []}
    train_cv_scores = {"WW": [], "HPC": [], "HPT": []}

    for i, test_esn in enumerate(unique_esns):
        print("\n" + "#" * 50)
        print(f"Fold {i + 1}/{len(unique_esns)}: Holding out ESN {test_esn}")
        print("#" * 50)

        train_mask = df["ESN"] != test_esn
        test_mask = df["ESN"] == test_esn

        # Raw Train/Test Splits
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()

        y_train = y[train_mask]
        y_test = y[test_mask]

        # --- Dimensionality Reduction Pipeline ---
        print("  Applying PCA Dimensionality Reduction...")
        reducer = SensorReducer(n_components=0.95)

        # Fit  on TRAIN, Transform TRAIN
        X_train_pca = reducer.fit_transform(df_train)

        # Transform TEST (using scaler/pca from train)
        X_test_pca = reducer.transform(df_test)

        print(f"  Reduced to {X_train_pca.shape[1]} components.")

        pred_dict = {}
        pred_train_dict = {}

        for target_name in TARGET_COLUMNS:
            # print(f"  Training {target_name}...")
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X_train_pca, y_train[target_name])
            pred_dict[target_name] = model.predict(X_test_pca)
            pred_train_dict[target_name] = model.predict(X_train_pca)

        # Evaluate Fold Test
        df_pred_test = pd.DataFrame(pred_dict)

        # Evaluate Fold Train
        df_pred_train = pd.DataFrame(pred_train_dict)

        # Calculate scores
        # WW
        true_WW = y_test["Cycles_to_WW"].values
        pred_WW = df_pred_test["Cycles_to_WW"].values
        score_WW = np.mean(
            time_weighted_error(
                true_WW, pred_WW, alpha=0.01, beta=1 / float(max(true_WW))
            )
        )
        cv_scores["WW"].append(score_WW)

        # WW Train
        true_WW_train = y_train["Cycles_to_WW"].values
        pred_WW_train = df_pred_train["Cycles_to_WW"].values
        score_WW_train = np.mean(
            time_weighted_error(
                true_WW_train,
                pred_WW_train,
                alpha=0.01,
                beta=1 / float(max(true_WW_train)),
            )
        )
        train_cv_scores["WW"].append(score_WW_train)

        # HPC
        true_HPC = y_test["Cycles_to_HPC_SV"].values
        pred_HPC = df_pred_test["Cycles_to_HPC_SV"].values
        score_HPC = np.mean(
            time_weighted_error(
                true_HPC, pred_HPC, alpha=0.01, beta=2 / float(max(true_HPC))
            )
        )
        cv_scores["HPC"].append(score_HPC)

        # HPC Train
        true_HPC_train = y_train["Cycles_to_HPC_SV"].values
        pred_HPC_train = df_pred_train["Cycles_to_HPC_SV"].values
        score_HPC_train = np.mean(
            time_weighted_error(
                true_HPC_train,
                pred_HPC_train,
                alpha=0.01,
                beta=2 / float(max(true_HPC_train)),
            )
        )
        train_cv_scores["HPC"].append(score_HPC_train)

        # HPT
        true_HPT = y_test["Cycles_to_HPT_SV"].values
        pred_HPT = df_pred_test["Cycles_to_HPT_SV"].values
        score_HPT = np.mean(
            time_weighted_error(
                true_HPT, pred_HPT, alpha=0.01, beta=2 / float(max(true_HPT))
            )
        )
        cv_scores["HPT"].append(score_HPT)

        # HPT Train
        true_HPT_train = y_train["Cycles_to_HPT_SV"].values
        pred_HPT_train = df_pred_train["Cycles_to_HPT_SV"].values
        score_HPT_train = np.mean(
            time_weighted_error(
                true_HPT_train,
                pred_HPT_train,
                alpha=0.01,
                beta=2 / float(max(true_HPT_train)),
            )
        )
        train_cv_scores["HPT"].append(score_HPT_train)

        print(
            f"  > Fold TEST Scores  - WW: {score_WW:.4f}, HPC: {score_HPC:.4f}, HPT: {score_HPT:.4f}"
        )
        # print(
        #     f"  > Fold TRAIN Scores - WW: {score_WW_train:.4f}, HPC: {score_HPC_train:.4f}, HPT: {score_HPT_train:.4f}"
        # )

    # Average Scores
    print("\n" + "=" * 50)
    print("📊 Average LOGO-CV Results (PCA Features)")
    print("=" * 50)

    # Train stats
    avg_WW_train = np.mean(train_cv_scores["WW"])
    avg_HPC_train = np.mean(train_cv_scores["HPC"])
    avg_HPT_train = np.mean(train_cv_scores["HPT"])
    final_score_train = np.mean([avg_WW_train, avg_HPC_train, avg_HPT_train])

    print(f"TRAIN Avg Score WW:  {avg_WW_train:.4f}")
    print(f"TRAIN Avg Score HPC: {avg_HPC_train:.4f}")
    print(f"TRAIN Avg Score HPT: {avg_HPT_train:.4f}")
    print(f"🏆 TRAIN Final Score: {final_score_train:.4f}")
    print("-" * 30)

    avg_WW = np.mean(cv_scores["WW"])
    avg_HPC = np.mean(cv_scores["HPC"])
    avg_HPT = np.mean(cv_scores["HPT"])

    print(f"TEST Avg Score WW:  {avg_WW:.4f} (std: {np.std(cv_scores['WW']):.4f})")
    print(f"TEST Avg Score HPC: {avg_HPC:.4f} (std: {np.std(cv_scores['HPC']):.4f})")
    print(f"TEST Avg Score HPT: {avg_HPT:.4f} (std: {np.std(cv_scores['HPT']):.4f})")

    final_score = np.mean([avg_WW, avg_HPC, avg_HPT])
    print(f"\n🏆 TEST Final CV Score: {final_score:.4f}")


if __name__ == "__main__":
    main()
