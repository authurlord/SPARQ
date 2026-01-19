import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, coercing errors (like '-') to NaN
df['Crude birth rate (per 1000)'] = pd.to_numeric(df['Crude birth rate (per 1000)'], errors='coerce')
df['Crude death rate (per 1000)'] = pd.to_numeric(df['Crude death rate (per 1000)'], errors='coerce')

# Calculate the difference
df['difference'] = df['Crude birth rate (per 1000)'] - df['Crude death rate (per 1000)']

# Find the year with the maximum positive difference
max_diff_year = df.loc[df['difference'].idxmax(), 'Unnamed: 0']

print(f"Final Answer: {max_diff_year}")