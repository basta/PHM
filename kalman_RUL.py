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

    # Set styling
    plt.style.use("seaborn-v0_8-darkgrid")
    return (
        LinearRegression,
        RandomForestRegressor,
        SimpleImputer,
        mo,
        np,
        pd,
        plt,
        train_test_split,
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
    def time_weighted_error(y_true, y_pred, alpha=0.02, beta=1):
        """
        Official Asymmetric Scoring Function.
        Late predictions (error >= 0) are penalized 2x more than early predictions.
        """
        # Ensure numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        error = y_pred - y_true

        # Weight calculation based on proximity to event (y_true)
        # and direction of error (late vs early)
        weight = np.where(
            error >= 0, 2 / (1 + alpha * y_true), 1 / (1 + alpha * y_true)
        )
        return weight * (error**2) * beta

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
    # We will define the sensor config here
    # Operating Condition Features (Inputs to Baseline Model)
    env_features = [
        "Sensed_WFuel",
        "Sensed_Altitude",
        "Sensed_Mach",
        "Sensed_Pamb",
        "Sensed_TAT",
    ]

    # Sensors to Monitor (Targets for Baseline -> Inputs for KF)
    # We chose sensors relevant to both Compressors (HPC) and Turbines (HPT)
    target_sensors = [
        "Sensed_T45",  # HPT Exit (Key for Hot Section)
        "Sensed_T3",  # HPC Exit (Key for Compressor)
        "Sensed_Ps3",  # HPC Static Pressure
        "Sensed_T5",  # EGT (General Health)
    ]
    return env_features, target_sensors


@app.cell
def _(LinearRegression, SimpleImputer, env_features, pd, target_sensors):
    # Data Processing Pipeline
    df_raw = pd.read_csv("training_data.csv")
    df_raw = df_raw.dropna()

    df_processed = df_raw.copy()

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
            model = LinearRegression()
            model.fit(baseline_data[env_features], baseline_data[sensor])

            # Predict "Expected" Value
            predicted = model.predict(df_processed[env_features])

            # Residual = Measured - Expected
            # This isolates degradation from operating conditions
            df_processed[f"Res_{sensor}"] = df_processed[sensor] - predicted

    df_processed
    return (df_processed,)


@app.cell
def _(np):
    # Robust Kalman Filter Implementation
    class RobustKalmanFilter:
        def __init__(self, dt=1.0, process_noise=1e-5, meas_noise=10.0):
            # State: [Health (Level), Degradation_Rate (Slope)]
            self.x = np.zeros((2, 1))
            self.P = np.eye(2) * 100.0
            self.F = np.array([[1.0, dt], [0.0, 1.0]])  # Constant Velocity
            self.H = np.array([[1.0, 0.0]])  # Measure Position only

            # Tuning
            self.Q = np.array(
                [
                    [process_noise, 0.0],
                    [0.0, process_noise * 0.01],  # Slope changes slower than position
                ]
            )
            self.R = np.array([[meas_noise]])

        def predict(self):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            return self.x.flatten()

        def update(self, z):
            y = z - (self.H @ self.x)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + (K @ y)
            self.P = (np.eye(2) - K @ self.H) @ self.P
            return self.x.flatten()

        def handle_maintenance(self, z_new, keep_slope=True):
            """
            Reset Health, but optionally keep knowledge of degradation rate.
            """
            old_slope = self.x[1, 0]
            self.x = np.zeros((2, 1))
            self.x[0, 0] = z_new
            if keep_slope:
                self.x[1, 0] = old_slope  # Assume degradation physics hasn't changed

            # Reset Variance to allow rapid convergence to new state
            self.P = np.eye(2) * 50.0
    return (RobustKalmanFilter,)


@app.cell
def _(RobustKalmanFilter, df_processed, pd, target_sensors):
    # Apply KF to Multiple Sensors
    # This creates our Feature Vector for the Regressor

    def process_multisensor_kf(df_engine):
        # Dictionary to store results for this engine
        kf_outputs = {}

        # Initialize one KF per sensor
        kfs = {
            s: RobustKalmanFilter(process_noise=1e-4, meas_noise=5.0)
            for s in target_sensors
        }

        # Track Maintenance to trigger resets
        prev_ww = df_engine.iloc[0]["Cumulative_WWs"]
        prev_hpc = df_engine.iloc[0]["Cumulative_HPC_SVs"]

        # Storage lists
        results = {s: {"health": [], "slope": []} for s in target_sensors}

        for idx, row in df_engine.iterrows():
            curr_ww = row["Cumulative_WWs"]
            curr_hpc = row["Cumulative_HPC_SVs"]

            # Detect Event
            is_maintenance = (curr_ww > prev_ww) or (curr_hpc > prev_hpc)

            for sensor in target_sensors:
                z = row[f"Res_{sensor}"]
                kf = kfs[sensor]

                kf.predict()

                if is_maintenance:
                    # Reset KF state to new measurement
                    kf.handle_maintenance(z)

                state = kf.update(z)
                results[sensor]["health"].append(state[0])
                results[sensor]["slope"].append(state[1])

            prev_ww = curr_ww
            prev_hpc = curr_hpc

        # Flatten into dataframe columns
        out_df = pd.DataFrame(index=df_engine.index)
        for sensor in target_sensors:
            out_df[f"KF_Health_{sensor}"] = results[sensor]["health"]
            out_df[f"KF_Slope_{sensor}"] = results[sensor]["slope"]

        return out_df

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
    score_submitted_result,
    target_sensors,
    train_test_split,
):
    # Train the RUL Regressors

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

        X = df_train_clean[feature_cols]
        y = df_train_clean[target_cols]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train 3 Separate Regressors
        # We use RandomForest because RUL degradation is often non-linear near end-of-life
        # and it handles interactions between 'Slope' and 'Health' well.

        y_pred_test = y_test.copy()

        for target in target_cols:
            print(f"Training model for {target}...")
            # N_estimators=50 for speed in demo, increase to 200+ for competition
            rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
            rf.fit(X_train, y_train[target])
            models[target] = rf

            y_pred_test[target] = rf.predict(X_test)

        # Evaluation using Official Metric
        scores = score_submitted_result(y_test, y_pred_test)

        scores_text = f"""
        ### Evaluation Results (Official Metric)
        | Target | Time-Weighted Score (Lower is better) |
        | :--- | :--- |
        | **Water Wash** | {scores.get("WW", 0):.4f} |
        | **HPC Visit** | {scores.get("HPC", 0):.4f} |
        | **HPT Visit** | {scores.get("HPT", 0):.4f} |
        | **FINAL SCORE** | **{np.mean(list(scores.values())):.4f}** |
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


if __name__ == "__main__":
    app.run()
