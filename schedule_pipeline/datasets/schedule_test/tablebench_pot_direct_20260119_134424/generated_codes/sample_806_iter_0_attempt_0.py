import pandas as pd

df = pd.read_csv('table.csv')
# Convert boiling point values to float
boiling_points = pd.to_numeric(df['boiling point (degree)'], errors='coerce')

# Calculate mean, median, and standard deviation
mean_bp = boiling_points.mean()
median_bp = boiling_points.median()
std_bp = boiling_points.std()

# Find the gas with the most extreme boiling point relative to the mean
deviations = abs(boiling_points - mean_bp)
most_extreme_gas = df.loc[deviations.idxmax(), 'helium']  # Using 'helium' column name as a placeholder; actual gas name is in index

# Since the first column is 'physical property', the gas names are in the first row
gas_names = df['physical property'].iloc[0]  # This is incorrect; instead, the gas names are in the first column's row data

# Correct approach: the gas names are in the first column of the data, but we need to map them properly
gas_names = df.columns[1:]  # Skip the 'physical property' column
boiling_points_with_gas = pd.DataFrame({'gas': gas_names, 'boiling_point': boiling_points})

# Find the gas with the most extreme boiling point
most_extreme_gas = boiling_points_with_gas.loc[boiling_points_with_gas['boiling_point'].idxmax()]['gas']

# Print results
print(f"Mean boiling point: {mean_bp:.2f}")
print(f"Median boiling point: {median_bp:.2f}")
print(f"Standard deviation: {std_bp:.2f}")
print(f"Gas with most extreme boiling point: {most_extreme_gas}")