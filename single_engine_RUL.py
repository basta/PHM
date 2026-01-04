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
    try:
        df_raw = pd.read_csv("training_data.csv")
        df_raw = df_raw.dropna()
    except FileNotFoundError:
        df_raw = pd.DataFrame()
    return (df_raw,)


@app.cell
def _(df_raw, mo):
    # Engine Selector
    if df_raw.empty:
        esn_select = mo.ui.md("**Error:** `training_data.csv` not found or empty.")
    else:
        esn_list = sorted(df_raw["ESN"].unique().tolist())
        esn_select = mo.ui.dropdown(esn_list, value=esn_list[0], label="Select ESN")

    mo.vstack([mo.md("### 1. Select Engine"), esn_select])
    return (esn_select,)


@app.cell
def _(
    Pipeline,
    PolynomialFeatures,
    Ridge,
    SimpleImputer,
    df_raw,
    env_features,
    esn_select,
    mo,
    process_multisensor_kf,
    target_sensors,
):
    # Process Single Engine
    if df_raw.empty or esn_select is None:
        mo.stop(True, "No data to process.")

    selected_esn = esn_select.value

    # 1. Filter
    df_engine = df_raw[df_raw["ESN"] == selected_esn].copy()
    df_engine = df_engine.sort_values("Cycles_Since_New")

    # 2. Preprocess (Ratios)
    epsilon = 1e-6
    if "Sensed_Ps3" in df_engine.columns and "Sensed_Pamb" in df_engine.columns:
        df_engine["Ratio_Ps3_Pamb"] = df_engine["Sensed_Ps3"] / (df_engine["Sensed_Pamb"] + epsilon)
    if "Sensed_T45" in df_engine.columns and "Sensed_TAT" in df_engine.columns:
        df_engine["Ratio_T45_TAT"] = df_engine["Sensed_T45"] / (df_engine["Sensed_TAT"] + 273.15)

    # 3. Impute
    # We use only this engine's data for imputation, fitting on itself.
    imputer = SimpleImputer(strategy="mean")
    try:
        df_engine[env_features] = imputer.fit_transform(df_engine[env_features])
    except ValueError:
        pass # Handle case where all are NaN if needed

    # 4. Baseline Residuals
    # Train on first 100 cycles of THIS engine
    N_healthy = 100
    baseline_data = df_engine[df_engine["Cycles_Since_New"] <= N_healthy]

    status_text = f"Processing ESN {selected_esn}: {len(df_engine)} cycles."

    if baseline_data.empty:
         status_text += f"\nWarning: Not enough data for baseline (<{N_healthy})."
    else:
        for sensor in target_sensors:
            if sensor not in df_engine.columns:
                continue

            poly = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('reg', Ridge(alpha=1.0))
            ])

            poly.fit(baseline_data[env_features], baseline_data[sensor])
            predicted = poly.predict(df_engine[env_features])
            df_engine[f"Res_{sensor}"] = df_engine[sensor] - predicted

        # 5. Kalman Filter
        # This function expects a DataFrame with Res_ cols.
        # It creates RobustKalmanFilter internally using defaults.
        kf_features = process_multisensor_kf(df_engine, noise=1e-4, meas_noise=200) 
        df_engine = df_engine.join(kf_features)
        status_text += "\nKalman Filter processing complete."

    mo.md(status_text)
    return (df_engine,)


@app.cell
def _(df_engine, mo, plt, target_sensors):
    # Plot KF States (Health)
    def plot_health(df):
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
                    # Calculate 1st and 99th percentiles to ignore top/bottom 1% extremes
                    q_low, q_high = series.quantile([0.01, 0.99])

                    # Handle flat lines (if min equals max, add default range)
                    if q_low == q_high:
                        margin = 0.1 if q_low == 0 else abs(q_low * 0.1)
                    else:
                        margin = (q_high - q_low) * 0.1  # Add 10% padding for visual clarity

                    ax2.set_ylim(q_low - margin, q_high + margin)

                # Combined Legend
                lines = l1 + l2
                labels = [l.get_label() for l in lines]
                axes[i].legend(lines, labels, loc="upper left")
            else:
                axes[i].legend(loc="upper left")

            axes[i].set_title(f"Sensor State: {sensor}")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    if not df_engine.empty and any(f"KF_Health_{s}" in df_engine.columns for s in target_sensors):
        health_plot = plot_health(df_engine)
        output = mo.vstack([mo.md("### 2. Health Indicator State Evolution"), health_plot])
    else:
        output = mo.md("No KF data generated.")

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
    df_engine,
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

    if df_engine.empty:
        mo.stop(True)

    def train_and_evaluate(df, models_dict, t_cols, f_cols, test_split=0.2):
        # Check data availability
        _useable_df = df.dropna(subset=t_cols + f_cols).copy()
        _useable_df = _useable_df.sort_values("Cycles_Since_New")

        _trained_models = {} 
        _model_scores = []

        if _useable_df.empty:
            return _useable_df, 0, _trained_models, _model_scores

        # Temporal Split
        n_total = len(_useable_df)
        n_test = int(n_total * test_split)
        n_train = n_total - n_test

        if n_train < 10:
             return _useable_df, 0, _trained_models, _model_scores

        train_df = _useable_df.iloc[:n_train]
        test_df = _useable_df.iloc[n_train:]
        split_cycle = train_df["Cycles_Since_New"].max()

        X_train = train_df[f_cols]
        y_train = train_df[t_cols]
        X_test = test_df[f_cols]
        y_test = test_df[t_cols]

        # UI Progress Info
        print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples (Split @ Cycle {split_cycle})...")

        for model_name, model_template in models_dict.items():
            _trained_models[model_name] = {}

            for target in t_cols:
                # Clone to ensure fresh start for each target
                _model = clone(model_template)

                # Fit
                _model.fit(X_train, y_train[target])
                _trained_models[model_name][target] = _model

                # Predict
                y_pred_train = _model.predict(X_train)
                y_pred_test = _model.predict(X_test)

                # Score
                _beta_val = 1.0
                if "WW" in target: _beta_val = 1.0 / np.max(y_train[target]) if np.max(y_train[target]) > 0 else 1
                elif "HPC" in target: _beta_val = 2.0 / np.max(y_train[target]) if np.max(y_train[target]) > 0 else 1
                elif "HPT" in target: _beta_val = 2.0 / np.max(y_train[target]) if np.max(y_train[target]) > 0 else 1

                # Avoid calculating score on empty arrays if something goes wrong, though n_train check handles most
                score_train = np.mean(time_weighted_error(y_train[target].values, y_pred_train, alpha=0.01, beta=_beta_val))
                score_test = np.mean(time_weighted_error(y_test[target].values, y_pred_test, alpha=0.01, beta=_beta_val))

                _model_scores.append({
                    "Model": model_name,
                    "Target": target,
                    "Train Score": score_train,
                    "Test Score": score_test
                })
        return _useable_df, split_cycle, _trained_models, _model_scores

    # Features
    feature_cols = []
    for _s in target_sensors:
        feature_cols.append(f"KF_Health_{_s}")
        feature_cols.append(f"KF_Slope_{_s}")

    useable_df, split_cycle_val, trained_models, model_scores = train_and_evaluate(df_engine, model_configs, target_cols, feature_cols)

    df_scores = pd.DataFrame(model_scores)

    if not df_scores.empty:
        # Pivot for nicer table - Complicated with 2 value cols, just melt or list
        # Simple list for now implies logic
        # Let's clean it up:
        df_display = df_scores.pivot(index="Model", columns="Target")
        # Flatten columns
        df_display.columns = [f"{c[1]} ({c[0]})" for c in df_display.columns]
        score_md = df_display.to_markdown()

        res_output = mo.md(f"### 3. Model Comparison (Train vs Test)\n\nTemporal Split at Cycle {split_cycle_val}\n\n{score_md}")
    else:
        res_output = mo.md("No scores computed.")

    res_output
    return (
        model_configs,
        split_cycle_val,
        target_cols,
        trained_models,
        useable_df,
    )


@app.cell
def _(mo, model_configs):
    # Model Selector
    model_names = list(model_configs.keys())
    model_select = mo.ui.dropdown(model_names, value="RandomForest", label="Select Model to Visualize")
    model_select
    return (model_select,)


@app.cell
def _(
    model_select,
    plt,
    split_cycle_val,
    target_cols,
    trained_models,
    useable_df,
):
    # Visualization of Selected Model
    selected_model_name = model_select.value

    def visualize_model(model_name, _df, _models, _targets, _split_cycle):
        if _df.empty or model_name not in _models:
            return None

        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        colors = ["blue", "green", "red"]

        # Re-predict using the stored model for this type
        X_viz = _df[_models[model_name][_targets[0]].feature_names_in_]

        for i, target in enumerate(_targets):
            model = _models[model_name][target]
            y_pred = model.predict(X_viz)
            y_true = _df[target]

            # Plot Truth
            axes[i].plot(_df["Cycles_Since_New"], y_true, 'k--', label="Truth")

            # Plot Prediction
            axes[i].plot(_df["Cycles_Since_New"], y_pred, color=colors[i], label=f"Pred ({model_name})")

            # Add Split Line
            axes[i].axvline(_split_cycle, color="gray", linestyle=":", label="Train/Test Split")

            # Add Shaded Background for Test Region
            axes[i].axvspan(_split_cycle, _df["Cycles_Since_New"].max(), color='gray', alpha=0.1, label="Test Region")

            axes[i].set_title(f"{target}")
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        return fig

    viz_output = visualize_model(selected_model_name, useable_df, trained_models, target_cols, split_cycle_val)
    viz_output
    return


if __name__ == "__main__":
    app.run()
