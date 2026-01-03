import argparse
import pandas as pd
import numpy as np
import time
from src.data import load_data, get_logo_cv_folds, get_feature_columns, TARGET_COLUMNS
from src.metrics import evaluate_fold
from src.features import SensorReducer, RollingFeaturesTransformer
from src.models import get_model


def main():
    parser = argparse.ArgumentParser(description="PHM RUL Estimation Training")
    parser.add_argument(
        "--model", type=str, default="xgboost", help="Model architecture"
    )
    parser.add_argument(
        "--use_pca", action="store_true", help="Use PCA dimensionality reduction"
    )
    parser.add_argument(
        "--pca_components",
        type=float,
        default=0.95,
        help="PCA explained variance ratio",
    )
    parser.add_argument(
        "--use_rolling",
        action="store_true",
        help="Generate rolling window features",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Rolling window size",
    )
    parser.add_argument(
        "--max_folds",
        type=int,
        default=None,
        help="Limit number of folds for debugging",
    )
    parser.add_argument(
        "--data_path", type=str, default="training_data_flat.csv", help="Path to data"
    )

    args = parser.parse_args()

    # 1. Load Data
    df = load_data(args.data_path)
    df = df.dropna()

    y = df[TARGET_COLUMNS]

    # Init Aggregators
    scores_history = []

    print(
        f"Starting LOGO-CV. Model: {args.model}, PCA: {args.use_pca}, Rolling: {args.use_rolling}"
    )

    folds = list(get_logo_cv_folds(df))
    total_folds = len(folds)

    for i, (train_mask, test_mask, test_esn) in enumerate(folds):
        if args.max_folds is not None and i >= args.max_folds:
            print(f"Stopping after {args.max_folds} folds (debug mode).")
            break

        print(f"\nFold {i + 1}/{total_folds}: ESN {test_esn}")

        # Split
        df_train_fold = df[train_mask].copy()
        df_test_fold = df[test_mask].copy()

        y_train_fold = y[train_mask]
        y_test_fold = y[test_mask]

        # Feature Engineering pipeline
        X_train_curr = df_train_fold
        X_test_curr = df_test_fold

        if args.use_rolling:
            rolling = RollingFeaturesTransformer(window_size=args.window_size)
            X_train_curr = rolling.fit_transform(X_train_curr)
            X_test_curr = rolling.transform(X_test_curr)

        if args.use_pca:
            # PCA uses "Sensed_" columns from the dataframe by default
            reducer = SensorReducer(n_components=args.pca_components)
            X_train_feat = reducer.fit_transform(X_train_curr)
            X_test_feat = reducer.transform(X_test_curr)
        else:
            # Uses all features defined (excluding metadata)
            # We recalculate feature columns to include any newly generated ones
            feat_cols = get_feature_columns(X_train_curr)
            X_train_feat = X_train_curr[feat_cols]
            X_test_feat = X_test_curr[feat_cols]

        # Per-Target Training
        pred_dict = {}
        pred_train_dict = {}

        for target in TARGET_COLUMNS:
            model = get_model(args.model)
            model.fit(X_train_feat, y_train_fold[target])
            pred_dict[target] = model.predict(X_test_feat)
            pred_train_dict[target] = model.predict(X_train_feat)

        # Evaluation - Test
        df_pred = pd.DataFrame(pred_dict, index=df_test_fold.index)
        fold_scores = evaluate_fold(y_test_fold, df_pred)

        # Evaluation - Train
        df_pred_train = pd.DataFrame(pred_train_dict, index=df_train_fold.index)
        fold_train_scores = evaluate_fold(y_train_fold, df_pred_train)

        # Merge and Store
        combined_scores = fold_scores.copy()
        for k, v in fold_train_scores.items():
            combined_scores[f"Train_{k}"] = v

        combined_scores["ESN"] = test_esn
        scores_history.append(combined_scores)

        print(
            f"  > TEST  Scores - WW: {fold_scores.get('WW', 0):.4f}, HPC: {fold_scores.get('HPC', 0):.4f}, HPT: {fold_scores.get('HPT', 0):.4f}"
        )
        print(
            f"  > TRAIN Scores - WW: {fold_train_scores.get('WW', 0):.4f}, HPC: {fold_train_scores.get('HPC', 0):.4f}, HPT: {fold_train_scores.get('HPT', 0):.4f}"
        )

    # Summary
    print("\n" + "=" * 30)
    print("AVERAGE METRICS")
    print("=" * 30)

    if scores_history:
        df_scores = pd.DataFrame(scores_history)
        mean_scores = df_scores.mean(numeric_only=True)
        std_scores = df_scores.std(numeric_only=True)

        print(mean_scores)
        print("\nStandard Deviations:")
        print(std_scores)
        print(f"\nOverall Final Score: {mean_scores.get('Final', 0):.4f}")
    else:
        print("No results.")


if __name__ == "__main__":
    main()
