import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2001
df_2001 = df[df['country'].str.contains('2001', na=False)]
# Find the country with the highest percentage of AIDS-related orphans
max_aids_orphans_country = df_2001.loc[df_2001['aids orphans as % of orphans'].idxmax()]
# Extract the percentage for Uganda in 2001
uganda_2001 = df_2001[df_2001['country'] == 'uganda (2001)']['aids orphans as % of orphans'].values[0]
# Print final answer
print(f"Final Answer: {max_aids_orphans_country['country'].split(' ')[0]}, {uganda_2001}")