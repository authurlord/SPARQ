import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the row with 'isaf exact total' since it contains non-numeric values
df_filtered = df[df['country'] != 'isaf exact total']

# Extract the two columns for correlation
troops_per_million = df_filtered['troops per one million population'].astype(float)
troops_per_gdp = df_filtered['troops per 1 billion ( usd ) gdp'].astype(float)

# Compute the correlation coefficient
correlation = troops_per_million.corr(troops_per_gdp)
print(f"Final Answer: {correlation:.3f}")