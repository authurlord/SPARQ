import pandas as pd

df = pd.read_csv('table.csv')
# Convert required columns to numeric
df['median household income'] = pd.to_numeric(df['median household income'])
df['population'] = pd.to_numeric(df['population'])

# Calculate correlation coefficient
correlation = df['median household income'].corr(df['population'])

print(f"Final Answer: {correlation:.4f}")