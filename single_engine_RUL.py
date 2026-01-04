import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNetCV

    from src.kalman_utils import env_features, target_sensors, process_multisensor_kf
    from src.metrics import time_weighted_error

    # Set styling
    plt.style.use("seaborn-v0_8-darkgrid")
    return (
        ElasticNetCV,
        GradientBoostingRegressor,
        LinearRegression,
        Pipeline,
        PolynomialFeatures,
        RandomForestRegressor,
        Ridge,
        SimpleImputer,
        StandardScaler,
        clone,
        env_features,
        mo,
        np,
        pd,
        plt,
        process_multisensor_kf,
        target_sensors,
        time_weighted_error,
    )


@app.cell
def _(mo):
    mo.md("""
    # ✈️ Single Engine RUL Analysis
    **Methodology:** Analyze one engine at a time from raw data to RUL prediction.
    """)
    return


@app.cell
def _(pd):

    # Load Data

    def clean_sensor_data(df):
        df_clean = df.copy()
        sensor_cols = [col for col in df.columns if col.startswith('Sensed_')]

        for col in sensor_cols:
            # 1. Physical Sanity Check: Remove negative values for non-altitude sensors
            if df_clean[col].dtype in [float, int] and 'Altitude' not in col:
                invalid_neg = df_clean[col] < 0
                if invalid_neg.sum() > 0:
                    df_clean = df_clean[~invalid_neg]

            # 2. Robust Statistical Outlier Removal (IQR)
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            # Using k=6.0 to preserve engine degradation signatures while removing massive errors
            lower_bound = Q1 - 6.0 * IQR
            upper_bound = Q3 + 6.0 * IQR

            is_outlier = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            if is_outlier.sum() > 0:
                df_clean = df_clean[~is_outlier]

        return df_clean

    try:
        df_raw = pd.read_csv("training_data.csv")
        df_raw = df_raw.dropna()
        df_raw = clean_sensor_data(df_raw)
    except FileNotFoundError:
        df_raw = pd.DataFrame()
    return (df_raw,)


@app.cell
def _(
    Pipeline,
    PolynomialFeatures,
    Ridge,
    SimpleImputer,
    StandardScaler,
    env_features,
    process_multisensor_kf,
    target_sensors,
):
    def preprocess_engine_data(df_raw, esn, imputer=None):
        """
        Processes a single engine from raw data to KF features.

        Args:
            df_raw: The complete raw dataframe.
            esn: The specific Engine Serial Number to process.
            imputer: If None, fits a new imputer (Train mode). 
                     If provided, uses it to transform (Test mode).
        """
        if df_raw.empty:
            return None, None

        # 1. Filter
        df = df_raw[df_raw["ESN"] == esn].copy()
        df = df.sort_values("Cycles_Since_New")

        if df.empty: 
            return None, None

        # 2. Preprocess (Ratios)
        epsilon = 1e-6
        if "Sensed_Ps3" in df.columns and "Sensed_Pamb" in df.columns:
            df["Ratio_Ps3_Pamb"] = df["Sensed_Ps3"] / (df["Sensed_Pamb"] + epsilon)
        if "Sensed_T45" in df.columns and "Sensed_TAT" in df.columns:
            df["Ratio_T45_TAT"] = df["Sensed_T45"] / (df["Sensed_TAT"] + 273.15)

        # 3. Impute
        # Crucial: If imputer is None, we are in 'Train' mode (fit). 
        # If imputer is passed, we are in 'Test' mode (transform only).
        if imputer is None:
            imputer = SimpleImputer(strategy="mean")
            df[env_features] = imputer.fit_transform(df[env_features])
        else:
            df[env_features] = imputer.transform(df[env_features])

        # 4. Baseline Residuals (Always self-referenced for residuals)
        # We assume an engine is healthy relative to ITSELF at the start.
        baseline_data = df[df["Cycles_Since_New"] <= 100]

        if len(baseline_data) < 10:
             return df, imputer # Return early if not enough data

        for sensor in target_sensors:
            if sensor not in df.columns: continue

            poly = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('reg', Ridge(alpha=1.0))
            ])
            # Fit baseline on the first 100 cycles of THIS engine
            poly.fit(baseline_data[env_features], baseline_data[sensor])
            predicted = poly.predict(df[env_features])
            df[f"Res_{sensor}"] = df[sensor] - predicted

        # 5. Kalman Filter
        kf_features = process_multisensor_kf(df, noise=1e-4, meas_noise=200) 
        df = df.join(kf_features)

        return df, imputer
    return (preprocess_engine_data,)


@app.cell
def _(df_raw, mo):
    if df_raw.empty:
        esn_select_train = mo.ui.md("No Data")
        esn_select_test = mo.ui.md("No Data")
    else:
        esn_list = sorted(df_raw["ESN"].unique().tolist())
        # Default to first and second engines
        esn_select_train = mo.ui.dropdown(esn_list, value=esn_list[0], label="Train Engine (Fit)")
        esn_select_test = mo.ui.dropdown(esn_list, value=esn_list[1] if len(esn_list) > 1 else esn_list[0], label="Test Engine (Evaluate)")

    mo.vstack([
        mo.md("### 1. Select Engines"), 
        mo.hstack([esn_select_train, esn_select_test])
    ])
    return esn_select_test, esn_select_train


@app.cell
def _(df_raw, esn_select_test, esn_select_train, mo, preprocess_engine_data):
    # 1. Process Training Engine
    if hasattr(esn_select_train, "value") and hasattr(esn_select_test, "value"):
         df_train, fitted_imputer = preprocess_engine_data(
            df_raw, 
            esn_select_train.value, 
            imputer=None # None = Fit the imputer on this data
        )

        # 2. Process Test Engine
        # Note: We pass the fitted_imputer to ensure we use training statistics
         df_test, _ = preprocess_engine_data(
            df_raw, 
            esn_select_test.value, 
            imputer=fitted_imputer 
        )

         if df_train is None or df_test is None:
             # mo.stop(True, "Error processing one of the engines.")
             status = mo.md("**Error:** Processing failed.")
         else:
             status = mo.md(f"**Pipeline Ready:** Trained on ESN {esn_select_train.value}, Testing on ESN {esn_select_test.value}")
    else:
         df_train, df_test = None, None
         status = mo.md("Waiting for selection...")

    status
    return df_test, df_train


@app.cell
def _(df_test, mo, plt, target_sensors):
    # Plot KF States (Health) for BOTH engines (optional, simpler to just show Test or both)
    # Let's show Test Engine Health to see what we are predicting on

    def plot_health(df, title_prefix=""):
        fig, axes = plt.subplots(len(target_sensors), 1, figsize=(10, 4*len(target_sensors)), sharex=True)
        if len(target_sensors) == 1:
            axes = [axes]

        for i, sensor in enumerate(target_sensors):
            health_col = f"KF_Health_{sensor}"
            slope_col = f"KF_Slope_{sensor}"

            # Plot Health (Primary Axis)
            if health_col in df.columns:
                l1 = axes[i].plot(df["Cycles_Since_New"], df[health_col], label="Health (Level)", color="purple")
                axes[i].set_ylabel("Health Level", color="purple")
                axes[i].tick_params(axis='y', labelcolor="purple")

            # Plot Slope (Secondary Axis)
            if slope_col in df.columns:
                ax2 = axes[i].twinx()
                l2 = ax2.plot(df["Cycles_Since_New"], df[slope_col], label="Degradation Rate (Slope)", color="orange", linestyle="--")
                ax2.set_ylabel("Slope", color="orange")
                ax2.tick_params(axis='y', labelcolor="orange")
                series = df[slope_col].dropna()
                if not series.empty:
                    q_low, q_high = series.quantile([0.01, 0.99])
                    if q_low == q_high:
                        margin = 0.1 if q_low == 0 else abs(q_low * 0.1)
                    else:
                        margin = (q_high - q_low) * 0.1
                    ax2.set_ylim(q_low - margin, q_high + margin)

                lines = l1 + l2
                labels = [l.get_label() for l in lines]
                axes[i].legend(lines, labels, loc="upper left")
            else:
                axes[i].legend(loc="upper left")

            axes[i].set_title(f"{title_prefix}Sensor State: {sensor}")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    output_list = [mo.md("### 2. Health Indicator State Evolution (Test Engine)")]
    if df_test is not None and not df_test.empty and any(f"KF_Health_{s}" in df_test.columns for s in target_sensors):
        output_list.append(plot_health(df_test, "Test Engine "))
    else:
        output_list.append(mo.md("No KF data for Test Engine."))

    output = mo.vstack(output_list)
    output
    return


@app.cell
def _(
    ElasticNetCV,
    GradientBoostingRegressor,
    LinearRegression,
    Pipeline,
    RandomForestRegressor,
    StandardScaler,
    clone,
    df_test,
    df_train,
    mo,
    np,
    pd,
    target_sensors,
    time_weighted_error,
):
    # Train Multi-Model Regressors & Evaluate

    target_cols = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]

    # Define Models
    model_configs = {
        "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        "LinearRegression": LinearRegression(),
        # "Ridge": Ridge(alpha=1.0),
        # "Lasso": Lasso(alpha=0.1),
        "ElasticNetCV": Pipeline([
            ('scaler', StandardScaler()),
            ('reg', ElasticNetCV(
                l1_ratio=[.1, .5, .7, .9, .95], 
                eps=0.001, 
                n_alphas=100, 
                cv=5, 
                n_jobs=-1,
                max_iter=5000
            ))
        ]),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
    }

    if df_train is None or df_test is None:
        mo.stop(True)

    def train_cross_engine(train_df, test_df, models_dict, t_cols, f_cols):
        # Clean NaNs
        train_df = train_df.dropna(subset=t_cols + f_cols).copy()
        test_df = test_df.dropna(subset=t_cols + f_cols).copy()

        if train_df.empty or test_df.empty:
            return {}, []

        _trained_models = {}
        _model_scores = []

        # Training Data
        X_train = train_df[f_cols]
        ys_train = train_df[t_cols]
        # Test Data
        X_test = test_df[f_cols]
        ys_test = test_df[t_cols]

        print(f"Training on {len(train_df)} samples, Testing on {len(test_df)} samples...")

        for model_name, model_template in models_dict.items():
            _trained_models[model_name] = {}

            for target in t_cols:
                _model = clone(model_template)

                # TRAIN on Engine A
                _model.fit(X_train, ys_train[target])
                _trained_models[model_name][target] = _model

                # PREDICT on Engine B (Test)
                y_pred_test = _model.predict(X_test)

                # Clip Predictions
                if "WW" in target:
                    y_pred_test = np.clip(y_pred_test, 0, 1200)
                elif "HPC" in target:
                    y_pred_test = np.clip(y_pred_test, 0, 10000)
                elif "HPT" in target:
                    y_pred_test = np.clip(y_pred_test, 0, 4500)

                # Score
                _beta_val = 1.0 
                if "WW" in target: _beta_val = 1.0 / np.max(ys_train[target]) if np.max(ys_train[target]) > 0 else 1
                elif "HPC" in target: _beta_val = 2.0 / np.max(ys_train[target]) if np.max(ys_train[target]) > 0 else 1
                elif "HPT" in target: _beta_val = 2.0 / np.max(ys_train[target]) if np.max(ys_train[target]) > 0 else 1

                score_test = np.mean(time_weighted_error(ys_test[target].values, y_pred_test, alpha=0.01, beta=_beta_val))

                _model_scores.append({
                    "Model": model_name,
                    "Target": target,
                    "Test Score": score_test
                })

        return _trained_models, _model_scores

    # Features
    feature_cols = []
    for _s in target_sensors:
        feature_cols.append(f"KF_Health_{_s}")
        feature_cols.append(f"KF_Slope_{_s}")

    trained_models, model_scores = train_cross_engine(df_train, df_test, model_configs, target_cols, feature_cols)

    if model_scores:
        df_scores = pd.DataFrame(model_scores)
        score_pivot = df_scores.pivot(index="Model", columns="Target", values="Test Score")
        score_md = score_pivot.to_markdown()
        res_output = mo.md(f"### 3. Cross-Engine Model Comparison\n\n**Test Scores (trained on A, tested on B):**\n\n{score_md}")
    else:
        res_output = mo.md("No scores computed (check data).")

    res_output
    return model_configs, target_cols, trained_models


@app.cell
def _(mo, model_configs):
    # Model Selector
    model_names = list(model_configs.keys())
    model_select = mo.ui.dropdown(model_names, value="RandomForest", label="Select Model to Visualize")
    model_select
    return (model_select,)


@app.cell
def _(df_test, model_select, np, plt, target_cols, trained_models):
    # Visualization of Selected Model
    selected_model_name = model_select.value

    def visualize_model_test(model_name, _df, _models, _targets):
        if _df is None or _df.empty or model_name not in _models:
            return None

        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        colors = ["blue", "green", "red"]

        # Re-predict using the stored model for this type
        X_viz = _df[_models[model_name][_targets[0]].feature_names_in_]

        for i, target in enumerate(_targets):
            model = _models[model_name][target]
            y_pred = model.predict(X_viz)

            # Clip Predictions
            if "WW" in target:
                y_pred = np.clip(y_pred, 0, 1200)
            elif "HPC" in target:
                y_pred = np.clip(y_pred, 0, 10000)
            elif "HPT" in target:
                y_pred = np.clip(y_pred, 0, 4500)

            y_true = _df[target]

            # Plot Truth
            axes[i].plot(_df["Cycles_Since_New"], y_true, 'k--', label="Truth (Test Engine)")

            # Plot Prediction
            axes[i].plot(_df["Cycles_Since_New"], y_pred, color=colors[i], label=f"Pred ({model_name})")

            # Add Split Line
            # axes[i].axvline(_split_cycle, color="gray", linestyle=":", label="Train/Test Split")

            # Add Shaded Background for Test Region
            # axes[i].axvspan(_split_cycle, _df["Cycles_Since_New"].max(), color='gray', alpha=0.1, label="Test Region")

            axes[i].set_title(f"{target}")
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        return fig

    viz_output = visualize_model_test(selected_model_name, df_test, trained_models, target_cols)
    viz_output
    return (selected_model_name,)


@app.cell
def _():
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

            self.streak = 0
            self.reset_threshold = 15

        def update(self, current_cycle, raw_rul_pred, reset=False):
            if reset:
                self.P = 10000.0

            # 1. Measurement: The ML model says EoL is (Now + Pred)
            z = current_cycle + raw_rul_pred

            # 2. Prediction Step (Random Walk: x_k = x_{k-1})
            # P increases slightly because we are uncertain about stability
            self.P = self.P + self.Q

            innovation = z - self.x

            # 3. Update Step
            K = self.P / (self.P + self.R)
            self.x = self.x + K * (z - self.x)
            self.P = (1 - K) * self.P

            # 4. Convert back to RUL
            smoothed_rul = self.x - current_cycle

            # Clip: RUL cannot be negative
            return max(0, smoothed_rul)

    def apply_smoothing(df, raw_pred_col, r_var, q_var):
        """Helper to apply smoother with dynamic tuning"""
        # Init state near the first prediction
        first_meas = df.iloc[0]["Cycles_Since_New"] + df.iloc[0][raw_pred_col]

        smoother = RULSmoother(r_variance=r_var, q_variance=q_var, initial_eol=first_meas)
        smoothed_vals = []

        # Event Detection Init
        # Check if columns exist to avoid errors if data is missing them
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
    return (apply_smoothing,)


@app.cell
def _(mo):
    mo.md("### 4. RUL Smoother Tuning (Log Scale)")

    r_log_slider = mo.ui.slider(
        start=1.0, stop=5.0, step=0.1, value=3.7, 
        label="Log10(R) - Measurement Noise"
    )


    q_log_slider = mo.ui.slider(
        start=-6.0, stop=2.0, step=0.1, value=0.0, 
        label="Log10(Q) - Process Noise"
    )

    # Display current actual values for clarity
    mo.vstack([
        r_log_slider, 
        q_log_slider,
    ])
    return q_log_slider, r_log_slider


@app.cell
def _(
    apply_smoothing,
    df_test,
    np,
    plt,
    q_log_slider,
    r_log_slider,
    selected_model_name,
    target_cols,
    trained_models,
):
    r_val = 10 ** r_log_slider.value
    q_val = 10 ** q_log_slider.value

    def visualize_model_test_smooth(model_name, _df, _models, _targets):
        if _df is None or _df.empty or model_name not in _models:
            return None

        # Create copy to avoid SettingWithCopy warnings on the main df
        plot_df = _df.copy()

        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Features for prediction
        X_viz = plot_df[_models[model_name][_targets[0]].feature_names_in_]

        for i, target in enumerate(_targets):
            model = _models[model_name][target]

            # 1. Raw Prediction
            raw_pred = model.predict(X_viz)

            # Clip Raw Predictions based on domain knowledge
            if "WW" in target: limit = 1200
            elif "HPC" in target: limit = 10000
            elif "HPT" in target: limit = 4500
            else: limit = 10000
            raw_pred = np.clip(raw_pred, 0, limit)

            plot_df[f"Pred_{target}"] = raw_pred

            # 2. Apply Smoothing (The new Logic)
            # We treat the Regressor output as a sensor measurement
            plot_df[f"Smooth_{target}"] = apply_smoothing(
                    plot_df, f"Pred_{target}", r_var=r_val, q_var=q_val
                )

            # 3. Plotting
            ax = axes[i]
            x_axis = plot_df["Cycles_Since_New"]

            # Truth
            ax.plot(x_axis, plot_df[target], 'k--', linewidth=1.5, label="Truth", alpha=0.6)

            # Raw ML Prediction
            ax.plot(x_axis, plot_df[f"Pred_{target}"], color="green", alpha=0.3, label=f"Raw ML ({model_name})")

            # Smoothed Prediction
            ax.plot(x_axis, plot_df[f"Smooth_{target}"], color="orange", linewidth=2.5, label="Filtered Prediction")

            ax.set_title(f"{target} Prediction")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Cycles Remaining")

        axes[-1].set_xlabel("Current Cycle Time")
        plt.tight_layout()
        return fig, plot_df

    viz_output_smoothed, smooth_plot_df = visualize_model_test_smooth(selected_model_name, df_test, trained_models, target_cols)
    viz_output_smoothed
    return (smooth_plot_df,)


@app.cell
def _(mo, np, smooth_plot_df, time_weighted_error):
    if smooth_plot_df is None or smooth_plot_df.empty:
        mo.stop(True)

    # Helper for safe access
    def get_score(target_col, beta_numerator):
        if target_col not in smooth_plot_df.columns or f"Smooth_{target_col}" not in smooth_plot_df.columns:
            return None

        y_true = smooth_plot_df[target_col].values
        y_pred = smooth_plot_df[f"Smooth_{target_col}"].values

        max_val = np.max(y_true)
        beta_val = beta_numerator / float(max_val) if max_val > 0 else 1.0

        score_arr = time_weighted_error(y_true, y_pred, alpha=0.01, beta=beta_val)
        return np.mean(score_arr)

    score_ww = get_score("Cycles_to_WW", 1.0)
    score_hpc = get_score("Cycles_to_HPC_SV", 2.0)
    score_hpt = get_score("Cycles_to_HPT_SV", 2.0)

    scores_final = {}
    if score_ww is not None: scores_final["WW"] = score_ww
    if score_hpc is not None: scores_final["HPC"] = score_hpc
    if score_hpt is not None: scores_final["HPT"] = score_hpt


    final_competition_score = np.mean([score_ww, score_hpc, score_hpt])

    mo.md(f"""
    ### 5. Final Competition Score (Smoothed)

    | Metric | Score |
    | :--- | :--- |
    | **Water Wash (WW)** | {score_ww:.6f} |
    | **HPC Shop Visit** | {score_hpc:.6f} |
    | **HPT Shop Visit** | {score_hpt:.6f} |
    | | |
    | **FINAL SCORE** | **{final_competition_score:.6f}** |
    """)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
