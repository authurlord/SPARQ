import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['total'] = pd.to_numeric(df['total'], errors='coerce')
df['population density (per km square)'] = pd.to_numeric(df['population density (per km square)'], errors='coerce')

# Calculate correlation coefficient
correlation = df['total'].corr(df['population density (per km square)'])
print(f"Final Answer: {correlation:.3f}")