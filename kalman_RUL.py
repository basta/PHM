import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import seaborn as sns
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import ElasticNetCV

    # Set styling
    plt.style.use("seaborn-v0_8-darkgrid")
    return (
        Pipeline,
        PolynomialFeatures,
        RandomForestRegressor,
        Ridge,
        SimpleImputer,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # ✈️ PHM North America 2025: Engine RUL Prediction
    **Methodology:** Multi-Sensor Kalman Filter Fusion + Regressors
    """)
    return


@app.cell
def _(np):
    from src.metrics import time_weighted_error

    def score_submitted_result(df_true, df_pred):
        """
        Aggregates scores for WW, HPC, and HPT targets.
        """
        scores = {}

        # 1. Water Wash (WW) Scoring
        if "Cycles_to_WW" in df_true.columns:
            true_WW = df_true.Cycles_to_WW.values
            pred_WW = df_pred.Cycles_to_WW.values
            scores["WW"] = np.mean(
                time_weighted_error(
                    true_WW, pred_WW, alpha=0.01, beta=1 / float(np.max(true_WW))
                )
            )

        # 2. HPC Shop Visit Scoring
        if "Cycles_to_HPC_SV" in df_true.columns:
            true_HPC = df_true.Cycles_to_HPC_SV.values
            pred_HPC = df_pred.Cycles_to_HPC_SV.values
            scores["HPC"] = np.mean(
                time_weighted_error(
                    true_HPC, pred_HPC, alpha=0.01, beta=2 / float(np.max(true_HPC))
                )
            )

        # 3. HPT Shop Visit Scoring
        if "Cycles_to_HPT_SV" in df_true.columns:
            true_HPT = df_true.Cycles_to_HPT_SV.values
            pred_HPT = df_pred.Cycles_to_HPT_SV.values
            scores["HPT"] = np.mean(
                time_weighted_error(
                    true_HPT, pred_HPT, alpha=0.01, beta=2 / float(np.max(true_HPT))
                )
            )

        return scores
    return (score_submitted_result,)


@app.cell
def _():
    from src.kalman_utils import env_features, target_sensors
    return env_features, target_sensors


@app.cell
def _(
    Pipeline,
    PolynomialFeatures,
    Ridge,
    SimpleImputer,
    env_features,
    pd,
    target_sensors,
):
    # Data Processing Pipeline
    df_raw = pd.read_csv("training_data.csv")
    df_raw = df_raw.dropna()

    epsilon = 1e-6

    df_processed = df_raw.copy()
    df_processed["Ratio_Ps3_Pamb"] = df_processed["Sensed_Ps3"] / (df_processed["Sensed_Pamb"] + epsilon)

    df_processed["Ratio_T45_TAT"] = df_processed["Sensed_T45"] / (df_processed["Sensed_TAT"] + 273.15)

    if not df_processed.empty:
        # 1. Sort
        df_processed = df_processed.sort_values(["ESN", "Cycles_Since_New"])

        # 2. Impute NaNs in Env Features
        imputer = SimpleImputer(strategy="mean")
        df_processed[env_features] = imputer.fit_transform(df_processed[env_features])

        # 3. Calculate Residuals for ALL Target Sensors
        # We train a baseline model on the first 100 cycles of ALL engines (assuming healthy start)
        baseline_data = df_processed[df_processed["Cycles_Since_New"] <= 100]

        residuals_dict = {}

        print("Calculating Baseline Residuals...")
        for sensor in target_sensors:
            # Fit model: Env Features -> Sensor Value
            # model = LinearRegression()
            # model.fit(baseline_data[env_features], baseline_data[sensor])

            # # Predict "Expected" Value
            # predicted = model.predict(df_processed[env_features])

            poly = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('reg', Ridge(alpha=1.0)) # Ridge is Linear Regression with L2 regularization
            ])

            poly.fit(baseline_data[env_features], baseline_data[sensor])
            predicted = poly.predict(df_processed[env_features])

            # Residual = Measured - Expected
            # This isolates degradation from operating conditions
            df_processed[f"Res_{sensor}"] = df_processed[sensor] - predicted

    df_processed
    return (df_processed,)


@app.cell
def _():
    from src.kalman_utils import RobustKalmanFilter
    return


@app.cell
def _(df_processed, pd):
    from src.kalman_utils import process_multisensor_kf

    # Apply KF to Multiple Sensors
    # This creates our Feature Vector for the Regressor

    if not df_processed.empty:
        print("Running Kalman Filters on all engines (this may take a moment)...")
        # Group by ESN and apply
        kf_features = df_processed.groupby("ESN").apply(process_multisensor_kf)
        kf_features = kf_features.reset_index(level=0, drop=True)  # Drop ESN index

        # Join back to main DF
        df_full = df_processed.join(kf_features)
    else:
        df_full = pd.DataFrame()

    df_full.head()
    return (df_full,)


@app.cell
def _(
    RandomForestRegressor,
    df_full,
    mo,
    np,
    pd,
    score_submitted_result,
    target_sensors,
):
    # Train the RUL Regressors using Leave-One-Group-Out (LOGO) CV

    # 1. Define Features (X)
    # We use the KF outputs for ALL sensors
    feature_cols = []
    for _s in target_sensors:
        feature_cols.append(f"KF_Health_{_s}")
        feature_cols.append(f"KF_Slope_{_s}")

    # 2. Define Targets (y)
    target_cols = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]

    # Check if we have data
    models = {}
    scores_text = "No data loaded."

    if not df_full.empty and all(c in df_full.columns for c in target_cols):
        # Drop rows where targets are NaN (if any)
        df_train_clean = df_full.dropna(subset=target_cols + feature_cols)
        unique_esns = sorted(df_train_clean["ESN"].unique())

        print(f"Starting LOGO-CV on {len(unique_esns)} engines...")

        fold_scores = []

        # -- LOCO Loop --
        for test_esn in unique_esns:
            train_mask = df_train_clean["ESN"] != test_esn
            test_mask = df_train_clean["ESN"] == test_esn

            X_train = df_train_clean.loc[train_mask, feature_cols]
            y_train = df_train_clean.loc[train_mask, target_cols]
            X_test = df_train_clean.loc[test_mask, feature_cols]
            y_test = df_train_clean.loc[test_mask, target_cols]

            # Train temporary models for this fold
            y_pred_fold = y_test.copy()
            for target in target_cols:
                rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
                rf.fit(X_train, y_train[target])
                y_pred_fold[target] = rf.predict(X_test)

            # Evaluate
            s = score_submitted_result(y_test, y_pred_fold)
            s["ESN"] = test_esn
            s["Final"] = np.mean(list(s.values()))
            fold_scores.append(s)

        # -- Aggregation --
        df_scores = pd.DataFrame(fold_scores).set_index("ESN")
        mean_scores = df_scores.mean(numeric_only=True)

        # -- Final Training (for Dashboard) --
        # We train on ALL data so the dashboard uses the best possible model
        X_all = df_train_clean[feature_cols]
        y_all = df_train_clean[target_cols]

        for target in target_cols:
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1)
            rf.fit(X_all, y_all[target])
            models[target] = rf

        # -- Reporting --
        scores_text = f"""
        ### LOGO-CV Evaluation Results (Mean over {len(unique_esns)} folds)

        | Target | Mean Score (Lower is better) | Std Dev |
        | :--- | :--- | :--- |
        | **Water Wash** | {mean_scores.get("WW", 0):.4f} | {df_scores["WW"].std():.4f} |
        | **HPC Visit** | {mean_scores.get("HPC", 0):.4f} | {df_scores["HPC"].std():.4f} |
        | **HPT Visit** | {mean_scores.get("HPT", 0):.4f} | {df_scores["HPT"].std():.4f} |
        | **FINAL SCORE** | **{mean_scores.get("Final", 0):.4f}** | **{df_scores["Final"].std():.4f}** |

        #### Per-Engine Breakdown

        {df_scores.round(4).to_markdown()}
        """

    mo.md(scores_text)
    return feature_cols, models, target_cols


@app.cell
def _(df_full, mo, models):
    # Interactive Dashboard - Selector
    if df_full.empty or not models:
        esn_select = None
    else:
        # UI Selection
        esn_list = sorted(df_full["ESN"].unique().tolist())
        esn_select = mo.ui.dropdown(esn_list, value=esn_list[0], label="Select ESN")
    return (esn_select,)


@app.cell
def _(df_full, esn_select, feature_cols, mo, models, plt, target_cols):
    # Interactive Dashboard - Plot
    if esn_select is None:
        plot_output = mo.md("Upload data and train models to see results.")
    else:

        def plot_prediction(selected_esn):
            df_plot = df_full[df_full["ESN"] == selected_esn].copy()

            # Generate Predictions
            X_plot = df_plot[feature_cols]

            fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

            colors = {
                "Cycles_to_WW": "blue",
                "Cycles_to_HPC_SV": "green",
                "Cycles_to_HPT_SV": "red",
            }

            for i, target in enumerate(target_cols):
                # Predict
                pred = models[target].predict(X_plot)

                # Plot Ground Truth
                ax[i].plot(
                    df_plot["Cycles_Since_New"],
                    df_plot[target],
                    label="Ground Truth",
                    color="black",
                    linestyle="--",
                    alpha=0.5,
                )

                # Plot Prediction
                ax[i].plot(
                    df_plot["Cycles_Since_New"],
                    pred,
                    label="Prediction",
                    color=colors[target],
                    linewidth=2,
                )

                # Plot Error Bounds (Visual aid)
                ax[i].fill_between(
                    df_plot["Cycles_Since_New"],
                    pred - 50,
                    pred + 50,
                    color=colors[target],
                    alpha=0.1,
                )

                ax[i].set_title(f"Prediction: {target}")
                ax[i].set_ylabel("Remaining Cycles")
                ax[i].legend()
                ax[i].grid(True, alpha=0.3)

            plt.xlabel("Cycles Since New")
            plt.tight_layout()
            return fig

        plot_output = mo.vstack([esn_select, plot_prediction(esn_select.value)])

    mo.vstack([mo.md("### 3. Engine Health Dashboard"), plot_output])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
