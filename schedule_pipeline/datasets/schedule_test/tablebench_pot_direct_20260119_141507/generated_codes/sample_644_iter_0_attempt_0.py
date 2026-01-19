import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract the relevant columns
boiling_point = df['boiling point (degree)'].astype(str).str.replace('-', np.nan).astype(float)
critical_temp = df['critical temperature (k)'].astype(float)

# Drop any rows where boiling point is NaN (due to '-')
boiling_point_clean = boiling_point.dropna()
critical_temp_clean = critical_temp[boiling_point_clean.index]

# Calculate correlation coefficient
correlation = np.corrcoef(boiling_point_clean, critical_temp_clean)[0, 1]
print(f"Final Answer: {correlation:.3f}")