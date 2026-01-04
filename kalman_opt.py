import optuna
import pandas as pd
import numpy as np
import argparse
from kalman_eval import evaluate_cv, Hyperparameters

def obj(trial):
    # Search Space
    hyp = Hyperparameters(
        # Preprocessing
        poly_degree=2,
        ridge_alpha=1.0, # Could opt, but keeping fixed for now
        
        # Kalman Filter
        kf_process_noise=trial.suggest_loguniform("kf_process_noise", 1e-6, 1e-2),
        kf_meas_noise=trial.suggest_loguniform("kf_meas_noise", 1.0, 1000.0),
        
        # RUL Smoother
        smoother_r_variance=trial.suggest_loguniform("smoother_r_variance", 100.0, 50000.0),
        smoother_q_variance=trial.suggest_loguniform("smoother_q_variance", 0.01, 100.0),
        smoother_reset_threshold=15,
        
        # Model
        model_type="RandomForest",
        rf_n_estimators=trial.suggest_int("rf_n_estimators", 10, 200),
        rf_max_depth=trial.suggest_int("rf_max_depth", 3, 20)
    )
    
    # Load Data
    try:
        df_raw = pd.read_csv("training_data.csv")
        df_raw = df_raw.dropna()
        # Note: cleaning is done inside evaluate_cv -> preprocess_engine if we added it there, 
        # but in kalman_eval.py current state, we added clean_sensor_data call in preprocess_engine 
        # (based on the user's manual edits).
        # We need to make sure we don't double clean or miss it.
        # kalman_eval.py loads, cleans, then calls evaluate_cv.
        # NO, wait. 
        # In kalman_eval.py:
        # main() -> loads df_raw -> evaluate_cv(df_raw...)
        # evaluate_cv -> calls preprocess_engine -> calls clean_sensor_data (as per user edit).
        # So we just pass raw dataframe here.
    except FileNotFoundError:
        return float('inf')

    # Run CV
    # We suppress prints to keep output clean, maybe?
    # For now let it print, it's fine.
    scores = evaluate_cv(df_raw, hyp, do_plots=False)
    
    if scores is None:
        return float('inf')
        
    # Minimize the mean of the 3 targets for simplicity, or just valid ones
    metrics = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]
    final_score = np.mean([scores[m] for m in metrics if m in scores])
    
    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--storage", type=str, default="sqlite:///kalman_opt.db")
    parser.add_argument("--study_name", type=str, default="kalman_rul_optimization")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name, 
        storage=args.storage, 
        direction="minimize",
        load_if_exists=True
    )
    
    study.optimize(obj, n_trials=args.trials)

    print("Best Params:", study.best_params)
