import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where county is not 'new mexico' (exclude state-level row)
df_filtered = df[df['county'] != 'new mexico']

# Extract 'median household income' and 'population' columns
income = df_filtered['median household income'].astype(float)
population = df_filtered['population'].astype(float)

# Calculate correlation coefficient
correlation = income.corr(population)

print(f"Final Answer: {correlation:.3f}")