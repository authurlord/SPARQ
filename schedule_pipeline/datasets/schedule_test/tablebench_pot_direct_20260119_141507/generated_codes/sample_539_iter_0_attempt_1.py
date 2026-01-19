import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows for years 1990 to 2000 (inclusive)
df_filtered = df[df['year'].str.contains(r'199[0-9]|200[0-9]')]

# Convert 'natural growth' to float and drop any na
df_filtered['natural growth'] = pd.to_numeric(df_filtered['natural growth'], errors='coerce')
df_filtered = df_filtered.dropna(subset=['natural growth'])

# Calculate average natural growth rate from 1990 to 2000
avg_growth = df_filtered['natural growth'].mean()

# Compute absolute deviations
df_filtered['deviation'] = abs(df_filtered['natural growth'] - avg_growth)

# Find the year with the maximum deviation
max_deviation_year = df_filtered.loc[df_filtered['deviation'].idxmax(), 'year']

print(f"Final Answer: {max_deviation_year}")