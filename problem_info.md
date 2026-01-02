# PHM North America 2025 Conference Data Challenge
**Domain:** Engine Health Management (EHM) for Commercial Jet Engines

## 1. Problem Overview
The objective is to develop a predictive maintenance model for commercial jet engines. The model must estimate the **Remaining Useful Life (RUL)**—specifically the time (in flight cycles)—until three distinct maintenance events occur.

The system monitors gas-path components (e.g., compressors, turbines) and external systems (controls, accessories) using sensor data collected across various flight phases (Idle, Takeoff, Cruise, etc.).

## 2. Prediction Targets
You must predict the remaining cycles for the following three events:

1.  **`Cycles_to_HPT_SV`**: Cycles until High Pressure Turbine Shop Visit (Typ. $\le$ 6000 cycles).
2.  **`Cycles_to_HPC_SV`**: Cycles until High Pressure Compressor Shop Visit (Typ. $\le$ 12,500 cycles).
3.  **`Cycles_to_WW`**: Cycles until HPC Water-Wash (Typ. $\le$ 1250 cycles).

## 3. Dataset Specifications
The dataset consists of time-series snapshots taken during various phases of flight.

### 3.1 Metadata (Columns A-F)
| Column | Type | Description |
| :--- | :--- | :--- |
| `ESN` | Integer | **Engine Serial Number**. Unique identifier for the engine (e.g., 101, 102). |
| `Cycles_Since_New` | Integer | Current age of the engine in flight cycles. |
| `Snapshot` | Integer | The snapshot index within the current flight cycle (1-8). |
| `Cumulative_WWs` | Integer | Total count of Water Wash events performed historically. |
| `Cumulative_HPC_SVs` | Integer | Total count of HPC Shop Visits performed historically. |
| `Cumulative_HPT_SVs` | Integer | Total count of HPT Shop Visits performed historically. |

### 3.2 Sensor Data (Columns G-V)
These columns represent the state of the engine at the specific snapshot time.

| Feature Name | Symbol | Description (Inferred) |
| :--- | :--- | :--- |
| `Sensed_Altitude` | $Alt$ | Flight Altitude |
| `Sensed_Mach` | $Ma$ | Flight Mach Number |
| `Sensed_Pamb` | $P_{amb}$ | Ambient Pressure |
| `Sensed_Pt2` | $P_{t2}$ | Total Pressure at Fan Inlet |
| `Sensed_TAT` | $TAT$ | Total Air Temperature |
| `Sensed_WFuel` | $W_f$ | Fuel Flow Rate |
| `Sensed_VAFN` | $VAFN$ | Variable Area Fan Nozzle position |
| `Sensed_VBV` | $VBV$ | Variable Bleed Valve position |
| `Sensed_Fan_Speed` | $N1$ | Fan Speed (Low Pressure Spool) |
| `Sensed_Core_Speed` | $N2$ | Core Speed (High Pressure Spool) |
| `Sensed_T25` | $T_{2.5}$ | HPC Inlet Temperature |
| `Sensed_T3` | $T_3$ | HPC Exit Temperature |
| `Sensed_Ps3` | $P_{s3}$ | HPC Exit Static Pressure |
| `Sensed_T45` | $T_{4.5}$ | HPT Exit / LPT Inlet Temperature |
| `Sensed_P25` | $P_{2.5}$ | HPC Inlet Pressure |
| `Sensed_T5` | $T_5$ | LPT Exit Temperature (EGT) |

### 3.3 Targets (Columns W-Y)
These are the ground truth values for training.
* `Cycles_to_WW`
* `Cycles_to_HPC_SV`
* `Cycles_to_HPT_SV`

### 3.4 Data Sample
```csv
ESN,Cycles_Since_New,Snapshot,Cumulative_WWs,...,Sensed_Fan_Speed,Sensed_Core_Speed,...,Cycles_to_WW
101,0,1,0,...,1900.28,21261.36,...,910
101,0,2,0,...,1903.00,21280.04,...,910
101,0,3,0,...,1851.61,21014.28,...,910
```

## 4. Evaluation & Scoring
The scoring metric is **asymmetric** and **time-weighted**.
* **Asymmetry:** Late predictions (predicting an event arrives later than it actually does) are penalized more heavily than early predictions.
* **Time-Weighting:** Errors made closer to the actual event time are penalized more heavily than errors made when the event is far away.

### Scoring Function (Python)
The final score is the mean of the scores for all three targets.

```python
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
    weight = np.where(
        error >= 0,
        2 / (1 + alpha * y_true),
        1 / (1 + alpha * y_true)
    )
    return weight * (error ** 2) * beta

def score_submitted_result(df_true, df_pred):
    """
    Aggregates scores for WW, HPC, and HPT targets.
    """
    # 1. Water Wash (WW) Scoring
    true_WW = df_true.Cycles_to_WW.values
    pred_WW = df_pred.Cycles_to_WW.values
    score_WW = np.mean(time_weighted_error(
        true_WW, pred_WW, alpha=0.01, beta=1/float(max(true_WW))
    ))

    # 2. HPC Shop Visit Scoring
    true_HPC = df_true.Cycles_to_HPC_SV.values
    pred_HPC = df_pred.Cycles_to_HPC_SV.values
    score_HPC = np.mean(time_weighted_error(
        true_HPC, pred_HPC, alpha=0.01, beta=2/float(max(true_HPC))
    ))

    # 3. HPT Shop Visit Scoring
    true_HPT = df_true.Cycles_to_HPT_SV.values
    pred_HPT = df_pred.Cycles_to_HPT_SV.values
    score_HPT = np.mean(time_weighted_error(
        true_HPT, pred_HPT, alpha=0.01, beta=2/float(max(true_HPT))
    ))

    # Final Score (Lower is better)
    return np.mean([score_WW, score_HPC, score_HPT])