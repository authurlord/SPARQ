import pandas as pd

df = pd.read_csv('table.csv')
# Extract tower division data and years
tower_division = df['tower division'].astype(int)
years = df['year'].astype(int)

# Calculate annual growth rates
growth_rates = []
for i in range(1, len(tower_division)):
    rate = (tower_division[i] - tower_division[i-1]) / tower_division[i-1]
    growth_rates.append(rate)

# Average growth rate
avg_growth_rate = sum(growth_rates) / len(growth_rates)

# Project population for 1881
last_population = tower_division[-1]
projected_population = last_population * (1 + avg_growth_rate)

print(f"Final Answer: {round(projected_population)}")