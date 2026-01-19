import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract tower division data
tower_data = df['tower division'].values
years = np.arange(1801, 1872)

# Linear regression: y = mx + b
# We want to predict y at x = 1881
x = years - 1801  # shift to start from 0
y = tower_data

# Calculate slope (m) and intercept (b)
m = np.polyfit(x, y, 1)[0]
b = np.polyfit(x, y, 1)[1]

# Project for year 1881
x_projected = 1881 - 1801
projected_population = m * x_projected + b

print(f"Final Answer: {int(projected_population)}")