import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2001
df_2001 = df[df['country'].str.endswith('(2001)')]
# Find the country with the highest percentage of AIDS-related orphans
max_country = df_2001.loc[df_2001['aids orphans as % of orphans'].idxmax()]
# Get Uganda's value
uganda_2001 = df_2001[df_2001['country'] == 'uganda (2001)']['aids orphans as % of orphans'].values[0]
# Output the result
print(f"Final Answer: {max_country['country']}, {uganda_2001}")