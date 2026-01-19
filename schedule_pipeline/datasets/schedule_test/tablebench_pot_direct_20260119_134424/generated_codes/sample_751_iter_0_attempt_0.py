import pandas as pd

df = pd.read_csv('table.csv')
# Extract tower division data
tower_division = df[['year', 'tower division']].astype(int)
# Calculate growth rate between consecutive years
growth_rates = []
for i in range(1, len(tower_division)):
    prev_pop = tower_division.iloc[i-1]['tower division']
    curr_pop = tower_division.iloc[i]['tower division']
    growth_rate = (curr_pop - prev_pop) / prev_pop
    growth_rates.append(growth_rate)

# Average annual growth rate
avg_growth_rate = sum(growth_rates) / len(growth_rates)

# Project population for 1881
last_pop = tower_division.iloc[-1]['tower division']
projected_pop_1881 = last_pop * (1 + avg_growth_rate)

print(f"Final Answer: {round(projected_pop_1881)}")