import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Select relevant columns for analysis
columns_to_check = ['periselene (km)', 'aposelene (km)', 'eccentricity', 'inclination (deg) (to moon equator)']

# Compute Z-scores for each column
z_scores = df[columns_to_check].apply(lambda x: np.abs((x - x.mean()) / x.std()))

# Find rows where any Z-score > 2 (outliers)
outlier_rows = z_scores.loc[z_scores > 2].any(axis=1)

# Get the original row indices and data
unusual_data_points = df[outlier_rows]

print("Unusual data points in terms of orbital characteristics:")
print(unusual_data_points)

# Final Answer: list the rows that exhibit unusual patterns
Final Answer: 1, 3, 6