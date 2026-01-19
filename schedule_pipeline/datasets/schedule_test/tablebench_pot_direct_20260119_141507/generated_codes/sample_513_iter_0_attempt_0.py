import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where year is 2011 and convert relevant columns to float
df_filtered = df[df['year'] == '2011'].copy()
df_filtered['wind power'] = pd.to_numeric(df_filtered['wind power'], errors='coerce')
df_filtered['total'] = pd.to_numeric(df_filtered['total'], errors='coerce')

# Drop rows with NaN in wind power or total
df_filtered = df_filtered.dropna(subset=['wind power', 'total'])

# Calculate wind energy share
df_filtered['wind_share'] = df_filtered['wind power'] / df_filtered['total']

# Find the country with the highest wind energy share
max_share_country = df_filtered.loc[df_filtered['wind_share'].idxmax(), 'country']

print(f"Final Answer: {max_share_country}")