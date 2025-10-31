import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import time

# --- 1. Configuration ---

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!           UPDATE YOUR FILENAME HERE           !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CSV_FILENAME = "./training_data.csv"

# Subsetting options (use one of the following)
# Option A: Limit the number of rows read (fast, simplest). Set to None to read all.
MAX_ROWS = 10000  # e.g., 100_000 rows; set None to disable

# Option B: Sample a fraction per chunk while reading (memory-friendly for huge files).
# Set CHUNK_SAMPLE_FRAC to a value between 0 and 1 (e.g., 0.1 for ~10%). Set to None to disable.
CHUNK_SAMPLE_FRAC = None
CHUNK_SIZE = 10000
RANDOM_STATE = 42

# Define your feature columns (all sensors, settings, and history)
# We exclude ESN (it's an ID) and the target columns
FEATURE_COLUMNS = [
    "Cycles_Since_New",
    "Snapshot",  # Including Snapshot as a feature
    "Cumulative_WWs",
    "Cumulative_HPC_SVs",
    "Cumulative_HPT_SVs",
    "Sensed_Altitude",
    "Sensed_Mach",
    "Sensed_Pamb",
    "Sensed_Pt2",
    "Sensed_TAT",
    "Sensed_WFuel",
    "Sensed_VAFN",
    "Sensed_VBV",
    "Sensed_Fan_Speed",
    "Sensed_Core_Speed",
    "Sensed_T25",
    "Sensed_T3",
    "Sensed_Ps3",
    "Sensed_T45",
    "Sensed_P25",
    "Sensed_T5",
]

# Define your three target columns
TARGET_COLUMNS = ["Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV"]

# --- 2. Load and Prepare Data ---
print(f"Loading data from {CSV_FILENAME}...")
try:
    if CHUNK_SAMPLE_FRAC is not None and 0 < CHUNK_SAMPLE_FRAC < 1:
        sampled_chunks = []
        for chunk in pd.read_csv(CSV_FILENAME, chunksize=CHUNK_SIZE):
            # Sample a fraction from each chunk deterministically
            sampled = chunk.sample(frac=CHUNK_SAMPLE_FRAC, random_state=RANDOM_STATE)
            sampled_chunks.append(sampled)
        df = pd.concat(sampled_chunks, ignore_index=True)
        print(
            f"Loaded sampled data: {len(df):,} rows (~{CHUNK_SAMPLE_FRAC * 100:.1f}% per chunk)."
        )
    else:
        read_kwargs = {}
        if MAX_ROWS is not None:
            read_kwargs["nrows"] = MAX_ROWS
        df = pd.read_csv(CSV_FILENAME, **read_kwargs)
        if MAX_ROWS is not None:
            print(f"Loaded first {len(df):,} rows (MAX_ROWS).")
        else:
            print(f"Loaded full dataset: {len(df):,} rows.")
except FileNotFoundError:
    print(f"Error: File not found at '{CSV_FILENAME}'")
    print(
        "Please make sure the file is in the same directory or provide the full path."
    )
    exit()

# Drop rows with any missing values for this quick test
df = df.dropna()

# Separate features (X)
X = df[FEATURE_COLUMNS]

MODEL_BLACKLIST = ["GaussianProcessRegressor", "KernelRidge", "SVR", "NuSVR"]

# --- 3. Run LazyRegressor for Each Target ---

# We will loop through each target and train a separate set of models
for target_name in TARGET_COLUMNS:
    print("\n" + "=" * 50)
    print(f"🚀 Running LazyPredict for target: {target_name}")
    print("=" * 50)

    # Select the current target column
    y = df[target_name]

    # Split the data for this target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize LazyRegressor
    # verbose=0 suppresses the individual model training output
    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
    )

    # Fit all models
    start_time = time.time()
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    end_time = time.time()

    print(f"✅ Finished in {end_time - start_time:.2f} seconds.")

    # Print the ranked list of models for this target
    print(f"\n--- Model Performance for {target_name} ---")
    print(models)
    print("\n\n")

print("All tasks complete.")
