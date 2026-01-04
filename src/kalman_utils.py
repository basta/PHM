
import numpy as np
import pandas as pd

# Operating Condition Features (Inputs to Baseline Model)
env_features = [
    "Sensed_WFuel",
    "Sensed_Altitude",
    "Sensed_Mach",
    "Sensed_Pamb",
    "Sensed_TAT",
]

# Sensors to Monitor (Targets for Baseline -> Inputs for KF)
target_sensors = [
    "Ratio_T45_TAT",  # HPT Exit (Key for Hot Section)
    "Sensed_T3",  # HPC Exit (Key for Compressor)
    "Ratio_Ps3_Pamb",  # HPC Static Pressure
    "Sensed_T5",  # EGT (General Health)
    "Sensed_T25",
    "Sensed_P25",
]


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

class RobustKalmanFilter:
    def __init__(self, dt=1.0, process_noise=1e-6, meas_noise=10.0):
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
    
        # Josephson Form Update (Numerically Stable)
        # P = (I - KH)P(I - KH)' + KRK'
        I_KH = np.eye(2) - (K @ self.H)
        self.P = (I_KH @ self.P @ I_KH.T) + (K @ self.R @ K.T)
    
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
        self.P = np.array([
    [50.0, 0.0],   # Position uncertainty
    [0.0, 0.001]   # Slope uncertainty (keep small!)
])

def process_multisensor_kf(df_engine, noise=1e-4, meas_noise=1.0):
    # Dictionary to store results for this engine
    kf_outputs = {}

    # Initialize one KF per sensor
    kfs = {
        s: RobustKalmanFilter(process_noise=noise, meas_noise=meas_noise)
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
