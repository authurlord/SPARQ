import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2001
df_2001 = df[df['country'].str.endswith('(2001)')]

# Find the country with the highest percentage of AIDS-related orphans
max_aids_orphans_country = df_2001.loc[df_2001['aids orphans as % of orphans'].idxmax()]
highest_percentage = max_aids_orphans_country['aids orphans as % of orphans']
country_with_highest = max_aids_orphans_country['country']

# Get Uganda's value for 2001
uganda_2001 = df_2001[df_2001['country'] == 'uganda (2001)']
uganda_percentage = uganda_2001['aids orphans as % of orphans'].values[0]

print(f"Final Answer: {country_with_highest}, {uganda_percentage}")