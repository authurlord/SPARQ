import pandas as pd

df = pd.read_csv('table.csv')
# Extract tower division population and years
tower_division = df['tower division'].astype(int)
years = df['year'].astype(int)

# Calculate the growth rate (linear approximation)
growth_rate = (tower_division.iloc[-1] - tower_division.iloc[0]) / (years.iloc[-1] - years.iloc[0])

# Project population for 1881
projected_population = tower_division.iloc[-1] + growth_rate * (1881 - years.iloc[-1])
print(f"Final Answer: {int(projected_population)}")