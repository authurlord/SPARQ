import pandas as pd

df = pd.read_csv('table.csv')
# Extract boiling point values
boiling_points = df[df['physical property'] == 'boiling point (degree)'].iloc[0, 1:]
# Convert to numeric
boiling_points = pd.to_numeric(boiling_points)
# Calculate mean, median, and standard deviation
mean_bp = boiling_points.mean()
median_bp = boiling_points.median()
std_bp = boiling_points.std()
# Find the gas with the most extreme boiling point (largest absolute deviation from mean)
deviations = abs(boiling_points - mean_bp)
most_extreme_gas = boiling_points.idxmax() if deviations.max() == deviations.max() else boiling_points.idxmin()
print(f"Mean: {mean_bp:.2f}, Median: {median_bp:.2f}, Standard Deviation: {std_bp:.2f}")
print(f"Most extreme boiling point: {most_extreme_gas}")