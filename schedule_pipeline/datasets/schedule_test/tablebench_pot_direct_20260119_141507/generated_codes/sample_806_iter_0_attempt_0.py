import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract boiling point values (second row in the data)
boiling_points = df.iloc[0][1:].astype(float)

# Calculate mean, median, and standard deviation
mean_bp = np.mean(boiling_points)
median_bp = np.median(boiling_points)
std_bp = np.std(boiling_points)

# Find the gas with the most extreme boiling point (largest absolute deviation from mean)
deviations = np.abs(boiling_points - mean_bp)
most_extreme_gas = df.columns[1:][np.argmax(deviations)]

print(f"Final Answer: mean={mean_bp:.1f}, median={median_bp:.1f}, std={std_bp:.2f}, most_extreme={most_extreme_gas}")