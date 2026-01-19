import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract population data for Tabriz (row 4)
tabriz_data = df.iloc[3]  # Row index 3 corresponds to Tabriz
years = [1956, 1966, 1976, 1986, 1996, 2006]
populations = [int(tabriz_data[str(year)]) for year in years]

# Convert to numpy arrays
years_array = np.array(years)
populations_array = np.array(populations)

# Fit a linear regression model
coefficients = np.polyfit(years_array, populations_array, 1)
trend_line = np.poly1d(coefficients)

# Project population for 2026
projected_population_2026 = trend_line(2026)

print(f"Final Answer: {int(projected_population_2026)}")