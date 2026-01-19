import pandas as pd

df = pd.read_csv('table.csv')

# Extract tower division data and years
tower_division = df['tower division'].astype(int)
years = df['year'].astype(int)

# Calculate the compound annual growth rate (CAGR)
initial_population = tower_division.iloc[0]
final_population = tower_division.iloc[-1]
num_years = years.iloc[-1] - years.iloc[0]

# CAGR formula: (final/initial)^(1/n) - 1
cagr = (final_population / initial_population) ** (1 / num_years) - 1

# Project population for 1881
projected_population = final_population * (1 + cagr)

print(f"Final Answer: {int(projected_population)}")