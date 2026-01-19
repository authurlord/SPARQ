import pandas as pd

df = pd.read_csv('table.csv')
# Extract the tower division data and corresponding years
years = df['year'].astype(int)
tower_division = df['tower division'].astype(int)

# Calculate the annual growth rate (slope) using linear regression
# We'll use the last year (1871) and the previous year (1861) to estimate the trend
# But we can also fit a line across all points for better accuracy
from numpy import polyfit
coefficients = polyfit(years, tower_division, 1)
growth_rate = coefficients[0]  # slope (annual increase)

# Project population for 1881
projected_population_1881 = tower_division.iloc[-1] + growth_rate
print(f"Final Answer: {int(projected_population_1881)}")