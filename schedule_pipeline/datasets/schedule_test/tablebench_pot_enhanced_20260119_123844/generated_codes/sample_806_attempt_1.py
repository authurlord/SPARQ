import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Extract boiling point values
boiling_points = df[df['physical property'] == 'boiling point (degree)'].iloc[0, 1:].astype(float)

# Calculate mean, median, and standard deviation
mean_bp = boiling_points.mean()
median_bp = boiling_points.median()
std_bp = boiling_points.std()

# Find the gas with the most extreme boiling point relative to the mean
abs_diff = np.abs(boiling_points - mean_bp)
most_extreme_gas = boiling_points.index[abs_diff.idxmax()]

print(f"Mean: {mean_bp:.2f}")
print(f"Median: {median_bp:.2f}")
print(f"Standard Deviation: {std_bp:.2f}")
print(f"Final Answer: {most_extreme_gas}")