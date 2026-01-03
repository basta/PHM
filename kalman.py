import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", layout_file="layouts/kalman.grid.json")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.impute import SimpleImputer
    return LinearRegression, SimpleImputer, mo, np, pd, plt


@app.cell
def _(features, pd, target):
    df = pd.read_csv("training_data.csv")
    df = df.dropna(subset=[target])

    # Sort and offset cycles to distinguish snapshots
    df = df.sort_values(["ESN", "Cycles_Since_New"])
    df["Cycles_Since_New"] = (
        df["Cycles_Since_New"]
        + df.groupby(["ESN", "Cycles_Since_New"]).cumcount() * 0.1
    )

    # Filter outliers from the entire df
    for col in [target] + features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return IQR, Q1, Q3, col, df


@app.cell
def _():
    features = [
        "Sensed_WFuel",
        "Sensed_Altitude",
        "Sensed_Mach",
        "Sensed_Pamb",
        "Sensed_TAT",
    ]

    target = "Sensed_T45"
    return features, target


@app.cell
def _(IQR, LinearRegression, Q1, Q3, SimpleImputer, col, df, features, target):
    df_pred = df.copy()
    df_healthy = df_pred[df_pred["Cycles_Since_New"] <= 200].copy()
    print(f"Training baseline on {len(df_healthy)} healthy snapshots.")

    model = LinearRegression()
    imputer = SimpleImputer(strategy="mean")
    df_healthy = df_healthy.dropna(subset=[target])

    # Filter outliers using IQR for target AND features
    for _col in [target] + features:
        _Q1 = df_healthy[_col].quantile(0.25)
        _Q3 = df_healthy[_col].quantile(0.75)
        _IQR = _Q3 - _Q1
        df_healthy = df_healthy[
            (df_healthy[col] >= Q1 - 1.5 * IQR) & (df_healthy[col] <= Q3 + 1.5 * IQR)
        ]

    # Fit imputer on healthy data and transform
    X_train = imputer.fit_transform(df_healthy[features])
    model.fit(X_train, df_healthy[target])

    # Predict what T45 *should* be for the entire dataset
    # This effectively normalizes the data for Altitude, Mach, and Throttle differences
    # Transform full dataset features using the same imputer
    X_full = imputer.transform(df_pred[features])
    df_pred["T45_Predicted"] = model.predict(X_full)

    # Calculate the Residual (Innovation)
    # z = y_measured - y_predicted
    df_pred["T45_Residual"] = df_pred[target] - df_pred["T45_Predicted"]

    # View the result
    df_pred[["Cycles_Since_New", "Sensed_T45", "T45_Predicted", "T45_Residual"]]
    return df_healthy, df_pred


@app.cell
def _(df_healthy, plt):
    plt.hist(df_healthy["Sensed_T45"], bins=50)
    plt.gca()
    return


@app.cell
def _(df, mo):
    # UI Controls
    esn_options = df["ESN"].unique().tolist()
    esn_select = mo.ui.dropdown(
        options=esn_options, value=esn_options[0], label="Select ESN"
    )

    min_cycle = int(df["Cycles_Since_New"].min())
    max_cycle = int(df["Cycles_Since_New"].max())
    max_cycle = 20000
    cycle_range = mo.ui.range_slider(
        start=min_cycle,
        stop=max_cycle,
        value=[min_cycle, max_cycle],
        step=1,
        label="Cycles Range",
        full_width=True,
    )

    mo.vstack([esn_select, cycle_range])
    return cycle_range, esn_select


@app.cell
def _(cycle_range, df_pred, esn_select, plt):
    # Filter data based on UI selection
    selected_esn = esn_select.value
    c_min, c_max = cycle_range.value

    df_plot = df_pred[
        (df_pred["ESN"] == selected_esn)
        & (df_pred["Cycles_Since_New"] >= c_min)
        & (df_pred["Cycles_Since_New"] <= c_max)
    ]

    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Raw Sensor Data
    ax[0].plot(
        df_plot["Cycles_Since_New"],
        df_plot["Sensed_T45"],
        label="Raw Sensed T45",
        color="gray",
        alpha=0.7,
    )
    ax[0].plot(
        df_plot["Cycles_Since_New"],
        df_plot["T45_Predicted"],
        label="Baseline Model",
        color="blue",
        linestyle="--",
    )
    ax[0].set_ylabel("Temp (K)")
    ax[0].set_title(f"Raw Data vs Healthy Baseline (ESN: {selected_esn})")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot 2: The Residual (The degradation signal)
    ax[1].plot(
        df_plot["Cycles_Since_New"],
        df_plot["T45_Residual"],
        label="Residual (Innovation)",
        color="red",
    )
    ax[1].axhline(0, color="black", linestyle="--", linewidth=1)
    ax[1].set_ylabel("Residual Error")
    ax[1].set_title("The Degradation Signal")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Plot 3: Cumulative Metrics
    ax[2].plot(
        df_plot["Cycles_Since_New"], df_plot["Cumulative_WWs"], label="Cumulative WWs"
    )
    ax[2].plot(
        df_plot["Cycles_Since_New"],
        df_plot["Cumulative_HPC_SVs"],
        label="Cumulative HPC SVs",
    )
    ax[2].plot(
        df_plot["Cycles_Since_New"],
        df_plot["Cumulative_HPT_SVs"],
        label="Cumulative HPT SVs",
    )
    ax[2].set_ylabel("Cumulative Count")
    ax[2].set_xlabel("Cycles Since New")
    ax[2].set_title("Cumulative Metrics")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    # mo.as_html(fig)
    return


@app.cell
def _(np):
    class EngineKalmanFilter:
        def __init__(self, dt=1.0, process_noise=1e-5, meas_noise=5.0):
            # State: [Health (Residual), Degradation_Rate]
            self.x = np.zeros((2, 1))

            # Covariance: Start with high uncertainty
            self.P = np.eye(2) * 100.0

            # State Transition (F): Constant Velocity Model
            self.F = np.array([[1.0, dt], [0.0, 1.0]])

            # Observation Matrix (H): We only measure Health (position), not Rate
            self.H = np.array([[1.0, 0.0]])

            # Process Noise (Q): How much can the "true" state wiggle?
            # Low Q = stiffer line, High Q = follows wiggles
            self.Q = np.array([[process_noise, 0.0], [0.0, process_noise * 0.1]])

            # Measurement Noise (R): How noisy is the residual?
            self.R = np.array([[meas_noise]])

        def predict(self):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            return self.x.flatten()

        def update(self, z):
            # Standard KF Update
            y = z - (self.H @ self.x)  # Innovation
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)

            self.x = self.x + (K @ y)
            I = np.eye(2)
            self.P = (I - K @ self.H) @ self.P

            return self.x.flatten()

        def handle_maintenance(self, z_new):
            """
            Call this when a Water Wash is detected.
            We reset the 'Position' to the new measurement,
            but we keep the 'Slope' (Velocity) because the degradation rate
            likely hasn't changed, just the absolute level.
            """
            # Reset position to new measurement, keep old slope
            self.x[0, 0] = z_new

            # Increase uncertainty in position, but maybe tighten slope?
            # Let's just reset P for safety to allow re-convergence
            self.P = np.array([[10.0, 0.0], [0.0, 1.0]])
    return (EngineKalmanFilter,)


@app.cell
def _(EngineKalmanFilter, df_pred, pd):
    def process_engine_kf(df_engine):
        # Initialize Filter
        # You might need to tune 'process_noise' if the line is too stiff or too wiggly
        kf = EngineKalmanFilter(process_noise=1e-5, meas_noise=10.0)

        smoothed_state = []
        slope_state = []
        uncertainty = []

        prev_ww = df_engine.iloc[0]["Cumulative_WWs"]

        for i, row in df_engine.iterrows():
            z = row["T45_Residual"]
            current_ww = row["Cumulative_WWs"]

            # 1. Predict Step
            kf.predict()

            # 2. Check for Maintenance Event (Water Wash)
            if current_ww > prev_ww:
                # We detected a wash! Reset the filter state to the new "clean" value
                kf.handle_maintenance(z)
                prev_ww = current_ww

            # 3. Update Step
            state = kf.update(z)

            smoothed_state.append(state[0])
            slope_state.append(state[1])
            uncertainty.append(kf.P[0, 0])  # Variance of position

        return smoothed_state, slope_state, uncertainty

    # Apply to the dataframe (Group by ESN first!)
    # We create new columns to store the KF results
    results = df_pred.groupby("ESN").apply(
        lambda x: pd.DataFrame(
            zip(*process_engine_kf(x)),
            columns=["KF_Health", "KF_Slope", "KF_Var"],
            index=x.index,
        )
    )

    results = results.reset_index(level=0, drop=True)

    # Merge back into original dataframe
    df_results = df_pred.join(results)
    return (df_results,)


@app.cell
def _(cycle_range, df_results, esn_select, mo, plt):
    def _():
        # Re-using your UI controls
        selected_esn = esn_select.value
        c_min, c_max = cycle_range.value

        df_plot = df_results[
            (df_results["ESN"] == selected_esn)
            & (df_results["Cycles_Since_New"] >= c_min)
            & (df_results["Cycles_Since_New"] <= c_max)
        ]

        fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot 1: The Tracking (Residual vs KF)
        ax[0].plot(
            df_plot["Cycles_Since_New"],
            df_plot["T45_Residual"],
            label="Noisy Residual",
            color="red",
            alpha=0.3,
        )
        ax[0].plot(
            df_plot["Cycles_Since_New"],
            df_plot["KF_Health"],
            label="Kalman Filter Estimate (State)",
            color="green",
            linewidth=2,
        )
        ax[0].set_title(f"State Estimation (ESN: {selected_esn})")
        ax[0].set_ylabel("Health Degradation")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        ax[0].set_ylim(0, 200)

        # Plot 2: The Slope (Degradation Rate)
        # This is what you use to predict RUL!
        ax[1].plot(
            df_plot["Cycles_Since_New"],
            df_plot["KF_Slope"],
            label="Estimated Degradation Rate",
            color="purple",
        )
        ax[1].set_title("Degradation Rate (Slope)")
        ax[1].set_ylabel("Degradation per Cycle")
        ax[1].axhline(0, color="black", linestyle="--")  # Slope should be positive
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        ax[1].set_ylim(-1, 1)

        # Plot 3: Maintenance Events (Reference)
        ax[2].plot(
            df_plot["Cycles_Since_New"], df_plot["Cumulative_WWs"], label="Water Washes"
        )
        ax[2].plot(
            df_plot["Cycles_Since_New"],
            df_plot["Cumulative_HPC_SVs"],
            label="High Pressures",
        )
        ax[2].plot(
            df_plot["Cycles_Since_New"],
            df_plot["Cumulative_HPT_SVs"],
            label="Low Pressures",
        )
        ax[2].set_title("Events")
        ax[2].legend()

        plt.tight_layout()
        return mo.as_html(fig)

    #_()
    return


if __name__ == "__main__":
    app.run()
