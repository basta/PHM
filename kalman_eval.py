import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone

from src.kalman_utils import env_features, target_sensors, process_multisensor_kf, clean_sensor_data
from src.metrics import time_weighted_error

# Set styling for plots
plt.style.use("seaborn-v0_8-darkgrid")

@dataclass
class Hyperparameters:
    # Preprocessing
    imputer_strategy: str = "mean"
    poly_degree: int = 2
    ridge_alpha: float = 1.0
    
    # Kalman Filter
    kf_process_noise: float = 1e-4
    kf_meas_noise: float = 200.0
    
    # RUL Smoother
    smoother_r_variance: float = 5000.0 # roughly 10^3.7
    smoother_q_variance: float = 1.0    # roughly 10^0
    smoother_reset_threshold: int = 15
    
    # Model
    model_type: str = "RandomForest" # Options: RandomForest, LinearRegression, ElasticNetCV, GradientBoosting
    rf_n_estimators: int = 50
    rf_max_depth: int = 5
    
    # Evaluation
    metrics_alpha: float = 0.01

class RULSmoother:
    """
    A scalar Kalman Filter to smooth RUL predictions by estimating
    the constant (or slowly drifting) 'End of Life' cycle.
    """
    def __init__(self, r_variance=5000, q_variance=10, initial_eol=1000):
        # State: The predicted End-of-Life (Cycle number)
        self.x = initial_eol 
        self.P = 10000.0  # High initial uncertainty

        # Parameters
        self.R = r_variance  # Measurement noise (Variance of the ML model)
        self.Q = q_variance  # Process noise (How much true EoL can shift)

    def update(self, current_cycle, raw_rul_pred, reset=False):
        if reset:
            self.P = 10000.0

        # 1. Measurement: The ML model says EoL is (Now + Pred)
        z = current_cycle + raw_rul_pred

        # 2. Prediction Step (Random Walk: x_k = x_{k-1})
        self.P = self.P + self.Q

        # 3. Update Step
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P

        # 4. Convert back to RUL
        smoothed_rul = self.x - current_cycle

        # Clip: RUL cannot be negative
        return max(0, smoothed_rul)

def preprocess_engine(df_raw, esn, hyp: Hyperparameters, imputer=None):
    """
    Processes a single engine from raw data to KF features.
    """
    if df_raw.empty:
        return None, None

    # 1. Filter
    df = df_raw[df_raw["ESN"] == esn].copy()
    df = df.sort_values("Cycles_Since_New")
    df = clean_sensor_data(df)
    df = df.dropna()

    if df.empty: 
        return None, None

    # 2. Preprocess (Ratios)
    epsilon = 1e-6
    if "Sensed_Ps3" in df.columns and "Sensed_Pamb" in df.columns:
        df["Ratio_Ps3_Pamb"] = df["Sensed_Ps3"] / (df["Sensed_Pamb"] + epsilon)
    if "Sensed_T45" in df.columns and "Sensed_TAT" in df.columns:
        df["Ratio_T45_TAT"] = df["Sensed_T45"] / (df["Sensed_TAT"] + 273.15)

    # 3. Impute
    if imputer is None:
        imputer = SimpleImputer(strategy=hyp.imputer_strategy)
        df[env_features] = imputer.fit_transform(df[env_features])
    else:
        df[env_features] = imputer.transform(df[env_features])

    # 4. Baseline Residuals
    baseline_data = df[df["Cycles_Since_New"] <= 100]

    if len(baseline_data) < 10:
         return df, imputer # Return early if not enough data

    for sensor in target_sensors:
        if sensor not in df.columns: continue

        poly = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=hyp.poly_degree, include_bias=False)),
            ('reg', Ridge(alpha=hyp.ridge_alpha))
        ])
        # Fit baseline on the first 100 cycles of THIS engine
        poly.fit(baseline_data[env_features], baseline_data[sensor])
        predicted = poly.predict(df[env_features])
        df[f"Res_{sensor}"] = df[sensor] - predicted

    # 5. Kalman Filter
    kf_features = process_multisensor_kf(df, noise=hyp.kf_process_noise, meas_noise=hyp.kf_meas_noise) 
    df = df.join(kf_features)

    return df, imputer

def train_model(X_train, y_train, hyp: Hyperparameters):
    """
    Trains a model based on hyperparameters.
    """
    if hyp.model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=hyp.rf_n_estimators, max_depth=hyp.rf_max_depth, random_state=42)
    elif hyp.model_type == "LinearRegression":
        model = LinearRegression()
    elif hyp.model_type == "ElasticNetCV":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('reg', ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95], eps=0.001, n_alphas=100, cv=5, n_jobs=-1))
        ])
    elif hyp.model_type == "GradientBoosting":
        model = GradientBoostingRegressor(n_estimators=hyp.rf_n_estimators, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {hyp.model_type}")
    
    model.fit(X_train, y_train)
    return model

def apply_smoothing(df, raw_pred_col, hyp: Hyperparameters):
    """Helper to apply smoother"""
    # Init state near the first prediction
    first_meas = df.iloc[0]["Cycles_Since_New"] + df.iloc[0][raw_pred_col]
    
    smoother = RULSmoother(r_variance=hyp.smoother_r_variance, 
                           q_variance=hyp.smoother_q_variance, 
                           initial_eol=first_meas)
    smoothed_vals = []
    
    cols_to_check = ["Cumulative_WWs", "Cumulative_HPC_SVs", "Cumulative_HPT_SVs"]
    valid_cols = [c for c in cols_to_check if c in df.columns]
    
    prev_vals = {c: df.iloc[0][c] for c in valid_cols}
    
    for idx, row in df.iterrows():
        do_reset = False
        for c in valid_cols:
            curr_val = row[c]
            if curr_val > prev_vals[c]:
                do_reset = True
            prev_vals[c] = curr_val
            
        s_rul = smoother.update(row["Cycles_Since_New"], row[raw_pred_col], reset=do_reset)
        smoothed_vals.append(s_rul)
        
    return smoothed_vals

def evaluate_cv(df_raw, hyp: Hyperparameters, do_plots=False, plot_dir="./plots", limit_folds=None):
    unique_esns = sorted(df_raw["ESN"].unique())
    
    eval_esns = unique_esns
    if limit_folds:
        eval_esns = unique_esns[:limit_folds]
        
    print(f"Starting Cross-Validation on {len(eval_esns)} folds out of {len(unique_esns)} total engines")
    
    if do_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    target_cols = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
    results = []
    
    for test_esn in eval_esns:
        train_esns = [e for e in unique_esns if e != test_esn]
        print(f"\nProcessing Fold: Test ESN={test_esn}, Train ESNs={train_esns}")
        
        # 1. Process Train Data
        train_dfs = []
        train_imputer = None
        
        # We need to process the first train engine to get the fitted imputer
        # Then use that imputer for subsequent train engines AND the test engine? 
        # Actually, single_engine_RUL.py implies we fit a fresh imputer for the "training" set.
        # But here our training set is composed of MULTIPLE engines.
        # A robust way: concat raw train data -> fit imputer -> transform.
        # BUT preprocess_engine does other things per engine (baseline).
        # Let's stick to the pattern: Fit imputer on the FIRST training engine, 
        # then reuse it? Or better: Fit imputer on ALL training raw data combined?
        # The single_engine_RUL.py logic:
        # df_train, fitted_imputer = preprocess_engine_data(df_raw, esn_select_train.value, imputer=None)
        # It fits on ONE engine. 
        # For N-1 training, we should likely fit the imputer on the concatenated training data.
        # However, `preprocess_engine` takes one ESN.
        # Let's fit the imputer on the concatenated raw data of training engines first.
        
        train_raw_subset = df_raw[df_raw["ESN"].isin(train_esns)]
        imputer = SimpleImputer(strategy=hyp.imputer_strategy)
        imputer.fit(train_raw_subset[env_features])
        
        for t_esn in train_esns:
            df_t, _ = preprocess_engine(df_raw, t_esn, hyp, imputer=imputer)
            if df_t is not None:
                train_dfs.append(df_t)
        
        if not train_dfs:
            print(f"Warning: No valid training data for Test ESN {test_esn}")
            continue
            
        df_train_all = pd.concat(train_dfs)
        
        # 2. Process Test Data
        df_test, _ = preprocess_engine(df_raw, test_esn, hyp, imputer=imputer)
        
        if df_test is None:
            print(f"Warning: No valid test data for Test ESN {test_esn}")
            continue
            
        # 3. Train Models
        feature_cols = []
        for _s in target_sensors:
            feature_cols.append(f"KF_Health_{_s}")
            feature_cols.append(f"KF_Slope_{_s}")
            
        trained_models = {}
        for target in target_cols:
            # Clean NaNs
            train_valid = df_train_all.dropna(subset=[target] + feature_cols)
            
            if train_valid.empty:
                print(f"  No valid training labels for {target}")
                continue
                
            model = train_model(train_valid[feature_cols], train_valid[target], hyp)
            trained_models[target] = model
            
        # 4. Evaluate & Predict
        if do_plots:
            fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            
        fold_scores = {}
        
        for i, target in enumerate(target_cols):
            if target not in trained_models:
                continue
            
            feature_valid_test = df_test.dropna(subset=feature_cols)
            if feature_valid_test.empty:
                continue
                
            # Raw Prediction
            raw_pred = trained_models[target].predict(feature_valid_test[feature_cols])
            
            # Clip
            if "WW" in target: limit = 1200
            elif "HPC" in target: limit = 10000
            elif "HPT" in target: limit = 4500
            else: limit = 10000
            raw_pred = np.clip(raw_pred, 0, limit)
            
            feature_valid_test[f"Pred_{target}"] = raw_pred
            
            # Apply Smoothing
            smoothed_pred = apply_smoothing(feature_valid_test, f"Pred_{target}", hyp)
            feature_valid_test[f"Smooth_{target}"] = smoothed_pred
            
            # Score
            y_true = feature_valid_test[target].values
            y_pred = np.array(smoothed_pred)
            
            # Filter NaNs in truth if any (though we assume truth exists for eval)
            mask = ~np.isnan(y_true)
            if np.sum(mask) == 0:
                continue
                
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            beta_val = 1.0
            if "WW" in target: beta_val = 1.0 / np.max(y_true) if np.max(y_true) > 0 else 1
            elif "HPC" in target: beta_val = 2.0 / np.max(y_true) if np.max(y_true) > 0 else 1
            elif "HPT" in target: beta_val = 2.0 / np.max(y_true) if np.max(y_true) > 0 else 1
            
            score_arr = time_weighted_error(y_true, y_pred, alpha=hyp.metrics_alpha, beta=beta_val)
            score = np.mean(score_arr)
            fold_scores[target] = score
            
            print(f"  {target}: {score:.4f}")
            
            # Plot
            if do_plots:
                ax = axes[i]
                ax.plot(feature_valid_test["Cycles_Since_New"], feature_valid_test[target], 'k--', label="Truth")
                ax.plot(feature_valid_test["Cycles_Since_New"], feature_valid_test[f"Pred_{target}"], color="green", alpha=0.3, label="Raw")
                ax.plot(feature_valid_test["Cycles_Since_New"], feature_valid_test[f"Smooth_{target}"], color="orange", label="Smoothed")
                ax.set_title(f"{target} (Score: {score:.4f})")
                ax.legend()
        
        results.append({"ESN": test_esn, **fold_scores})
        
        if do_plots:
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f"eval_{test_esn}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  Saved plot to {plot_path}")

    # Aggregate Results
    if not results:
        print("No results generated.")
        return
        
    df_res = pd.DataFrame(results)
    print("\n=== Final Cross-Validation Results ===")
    print(df_res.to_markdown(index=False))
    
    mean_scores = df_res.mean(numeric_only=True)
    print("\n=== Average Scores ===")
    print(mean_scores.to_markdown())
    
    return mean_scores

def main():
    parser = argparse.ArgumentParser(description="Kalman Filter RUL Evaluation")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--plot_dir", type=str, default="./plots", help="Directory to save plots")
    parser.add_argument("--data", type=str, default="training_data.csv", help="Path to training data")
    
    # Expose some hyperparameters as CLI args for quick sweeps
    parser.add_argument("--rf_n_estimators", type=int, default=50)
    parser.add_argument("--rf_max_depth", type=int, default=5)
    
    # Kalman Filter params
    parser.add_argument("--kf_process_noise", type=float, default=1e-4)
    parser.add_argument("--kf_meas_noise", type=float, default=200.0)

    # Smoother params
    parser.add_argument("--smoother_r_variance", type=float, default=5000.0)
    parser.add_argument("--smoother_q_variance", type=float, default=1.0)
    
    args = parser.parse_args()
    
    hyp = Hyperparameters(
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        kf_process_noise=args.kf_process_noise,
        kf_meas_noise=args.kf_meas_noise,
        smoother_r_variance=args.smoother_r_variance,
        smoother_q_variance=args.smoother_q_variance
    )
    
    # Load Data
    try:
        df_raw = pd.read_csv(args.data)
        df_raw = df_raw.dropna() # Basic dropna
    except FileNotFoundError:
        print(f"Error: File {args.data} not found.")
        return

    evaluate_cv(df_raw, hyp, do_plots=args.plots, plot_dir=args.plot_dir)

if __name__ == "__main__":
    main()
