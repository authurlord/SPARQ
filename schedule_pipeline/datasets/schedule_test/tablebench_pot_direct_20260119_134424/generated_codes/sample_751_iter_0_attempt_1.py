import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract tower division and year data
years = df['year'].astype(int)
tower_division = df['tower division'].astype(int)

# Fit a linear model to estimate growth rate
slope, intercept = np.polyfit(years, tower_division, 1)

# Project population for 1881
projected_1881 = slope * 1881 + intercept

print(f"Final Answer: {int(projected_1881)}")