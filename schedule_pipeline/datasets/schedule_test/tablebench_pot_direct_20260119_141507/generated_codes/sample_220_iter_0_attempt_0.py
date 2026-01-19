import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years between 1959 and 1989 inclusive
filtered_df = df[(df['year (january)'].between(1959, 1989))]

# Extract population (in thousands) and urban percentage
population = filtered_df['population (000)'].astype(float)
urban_percent = filtered_df['urban , %'].astype(float)

# Calculate correlation
correlation = population.corr(urban_percent)

print(f"Final Answer: {correlation:.2f}")