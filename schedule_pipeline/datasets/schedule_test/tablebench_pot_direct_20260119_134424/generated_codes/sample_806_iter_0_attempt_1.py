import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract boiling point values and convert to float
boiling_points = df[df['physical property'] == 'boiling point (degree)'].iloc[0, 1:].astype(float)

# Calculate mean, median, and standard deviation
mean_bp = boiling_points.mean()
median_bp = boiling_points.median()
std_bp = boiling_points.std()

# Find the gas with the most extreme boiling point relative to the mean
deviations = np.abs(boiling_points - mean_bp)
most_extreme_gas = boiling_points.index[deviations.idxmax()]

print(f"Mean boiling point: {mean_bp:.2f}")
print(f"Median boiling point: {median_bp:.2f}")
print(f"Standard deviation: {std_bp:.2f}")
print(f"Gas with most extreme boiling point: {most_extreme_gas}")

# Final Answer: mean, median, std, and the gas with most extreme boiling point
print(f"Final Answer: {mean_bp:.2f}, {median_bp:.2f}, {std_bp:.2f}, {most_extreme_gas}")