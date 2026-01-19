import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' and 'population density (per km square)' to numeric
df['total'] = pd.to_numeric(df['total'], errors='coerce')
df['population density (per km square)'] = pd.to_numeric(df['population density (per km square)'], errors='coerce')

# Drop rows with NaN values in the relevant columns
df.dropna(subset=['total', 'population density (per km square)'], inplace=True)

# Calculate the correlation coefficient
correlation = df['total'].corr(df['population density (per km square)'])

print(f"Final Answer: {correlation:.4f}")