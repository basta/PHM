import numpy as np


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



def evaluate_fold(df_true, df_pred):
    """
    Evaluates a dictionary of predictions against true values.
    Expects df_true and df_pred to have columns: Cycles_to_WW, Cycles_to_HPC_SV, Cycles_to_HPT_SV
    """
    scores = {}

    # 1. Water Wash (WW) Scoring
    if "Cycles_to_WW" in df_pred.columns and "Cycles_to_WW" in df_true.columns:
        true_WW = df_true["Cycles_to_WW"].values
        pred_WW = df_pred["Cycles_to_WW"].values
        # Beta logic for WW is 1 / max
        beta_val = 1.0 / float(np.max(true_WW)) if np.max(true_WW) > 0 else 1.0

        score_WW = np.mean(
            time_weighted_error(true_WW, pred_WW, alpha=0.01, beta=beta_val)
        )
        scores["WW"] = score_WW

    # 2. HPC Shop Visit Scoring
    if "Cycles_to_HPC_SV" in df_pred.columns and "Cycles_to_HPC_SV" in df_true.columns:
        true_HPC = df_true["Cycles_to_HPC_SV"].values
        pred_HPC = df_pred["Cycles_to_HPC_SV"].values
        # Beta logic for HPC is 2 / max
        beta_val = 2.0 / float(np.max(true_HPC)) if np.max(true_HPC) > 0 else 1.0

        score_HPC = np.mean(
            time_weighted_error(true_HPC, pred_HPC, alpha=0.01, beta=beta_val)
        )
        scores["HPC"] = score_HPC

    # 3. HPT Shop Visit Scoring
    if "Cycles_to_HPT_SV" in df_pred.columns and "Cycles_to_HPT_SV" in df_true.columns:
        true_HPT = df_true["Cycles_to_HPT_SV"].values
        pred_HPT = df_pred["Cycles_to_HPT_SV"].values
        # Beta logic for HPT is 2 / max
        beta_val = 2.0 / float(np.max(true_HPT)) if np.max(true_HPT) > 0 else 1.0

        score_HPT = np.mean(
            time_weighted_error(true_HPT, pred_HPT, alpha=0.01, beta=beta_val)
        )
        scores["HPT"] = score_HPT

    # Final Score
    if len(scores) == 3:
        final_score = np.mean([scores["WW"], scores["HPC"], scores["HPT"]])
        scores["Final"] = final_score

    return scores
