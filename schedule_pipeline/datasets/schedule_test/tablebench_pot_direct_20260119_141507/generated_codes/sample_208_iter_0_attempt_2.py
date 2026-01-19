import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant columns
crude_birth_rate = df['Crude birth rate (per 1000)']
natural_change = df['Natural change (per 1000)']

# Calculate correlation
correlation = crude_birth_rate.corr(natural_change)

print(f"Final Answer: {correlation:.2f}")