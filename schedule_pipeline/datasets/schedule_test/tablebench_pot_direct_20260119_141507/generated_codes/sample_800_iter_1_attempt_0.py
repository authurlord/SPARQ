import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where language is not French
non_french_points = df[df['language'] != 'french']['points']
# Convert to numeric, coercing errors to NaN, then drop NaN
non_french_points_numeric = pd.to_numeric(non_french_points, errors='coerce')
# Drop NaN values and compute median
median_points = non_french_points_numeric.dropna().median()
print(f"Final Answer: {median_points}")