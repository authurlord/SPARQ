import pandas as pd

df = pd.read_csv('table.csv')
# Extract boiling point values
boiling_points = df[df['physical property'] == 'boiling point (degree)'].iloc[0, 1:].astype(float)

# Calculate mean, median, and standard deviation
mean_bp = boiling_points.mean()
median_bp = boiling_points.median()
std_bp = boiling_points.std()

# Find the gas with the most extreme boiling point (largest deviation from mean)
deviations = abs(boiling_points - mean_bp)
most_extreme_gas = deviations.idxmax()

print(f"Final Answer: {mean_bp:.1f}, {median_bp:.1f}, {std_bp:.1f}, {most_extreme_gas}")