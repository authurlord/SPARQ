import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for year 2011
df_2011 = df[df['year'] == 2011]

# Convert columns to numeric, errors='coerce' will convert non-numeric to NaN
df_2011['wind power'] = pd.to_numeric(df_2011['wind power'], errors='coerce')
df_2011['biomass and waste'] = pd.to_numeric(df_2011['biomass and waste'], errors='coerce')
df_2011['solar'] = pd.to_numeric(df_2011['solar'], errors='coerce')
df_2011['total'] = pd.to_numeric(df_2011['total'], errors='coerce')

# Calculate Renewable Energy Share
df_2011['renewable_share'] = (df_2011['wind power'] + df_2011['biomass and waste'] + df_2011['solar']) / df_2011['total']

# Find the country with the highest renewable share
max_share_country = df_2011.loc[df_2011['renewable_share'].idxmax(), 'country']

print(f"Final Answer: {max_share_country}")