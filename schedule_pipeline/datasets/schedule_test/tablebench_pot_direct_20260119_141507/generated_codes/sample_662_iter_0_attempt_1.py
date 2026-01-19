import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between the two specified columns
correlation = df['troops per one million population'].corr(df['troops per 1 billion ( usd ) gdp'])
print(f"Final Answer: {correlation:.3f}")