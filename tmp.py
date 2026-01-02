import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("training_data.csv")

# 1. General Overview
print("Dataset Shape:", df.shape)
print(df.describe())  # Check min/max to spot outliers (e.g., negative pressure)

# 2. Correlation Matrix: The "Thermodynamic Check"
# We want to see which sensors move together.
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
corr = df[
    ["Sensed_Altitude", "Sensed_Fan_Speed", "Sensed_T5", "Cycles_to_HPT_SV"]
].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Sensor & Target Correlation Matrix")
plt.show()

# 3. Operating Point Dependency (The Noise Problem)
# Visualizing how Altitude affects EGT (T5).
# If this scatter is messy, we can't use raw T5 for prediction directly.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Sensed_Altitude", y="Sensed_T5", alpha=0.5)
plt.title("Impact of Altitude on Exhaust Gas Temperature (T5)")
plt.xlabel("Altitude (ft)")
plt.ylabel("EGT (T5)")
plt.grid(True)
plt.show()

# 4. Trajectory Analysis (The Signal)
# Plotting the lifecycle of a SINGLE engine (ESN 101).
# We expect T5 to rise slightly as the engine degrades (cycles increase).
engine_101 = df[df["ESN"] == 101].sort_values("Cycles_Since_New")

plt.figure(figsize=(12, 5))
plt.plot(
    engine_101["Cycles_Since_New"], engine_101["Sensed_T5"], label="Raw T5", alpha=0.6
)

# Adding a rolling mean to smooth out the flight-phase noise
# This simulates a "Low Pass Filter" to see the degradation trend
plt.plot(
    engine_101["Cycles_Since_New"],
    engine_101["Sensed_T5"].rolling(window=20).mean(),
    color="red",
    linewidth=2,
    label="Smoothed T5 (Trend)",
)

plt.title("Engine 101: Exhaust Gas Temperature over Time")
plt.xlabel("Cycles Since New")
plt.ylabel("Temperature (T5)")
plt.legend()
plt.show()
