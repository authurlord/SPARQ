import pandas as pd

df = pd.read_csv('table.csv')
# Remove the last row which is "isaf exact total" as it represents an aggregate
df_filtered = df.drop(df.index[-1])

# Select the two columns of interest
troops_per_million = df_filtered['troops per one million population']
troops_per_gdp = df_filtered['troops per 1 billion ( usd ) gdp']

# Calculate the correlation coefficient
correlation = troops_per_million.corr(troops_per_gdp)
print(f"Final Answer: {correlation:.3f}")