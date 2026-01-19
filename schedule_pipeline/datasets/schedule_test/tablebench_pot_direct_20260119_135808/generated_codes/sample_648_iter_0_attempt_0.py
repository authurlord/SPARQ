import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['revenue (millions)'] = pd.to_numeric(df['revenue (millions)'], errors='coerce')
df['profit (millions)'] = pd.to_numeric(df['profit (millions)'], errors='coerce')

# Calculate correlation coefficient
correlation = df['revenue (millions)'].corr(df['profit (millions)'])
print(f"Final Answer: {correlation:.4f}")