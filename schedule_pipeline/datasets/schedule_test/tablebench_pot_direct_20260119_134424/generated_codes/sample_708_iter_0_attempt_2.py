import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for year 2001
df_2001 = df[df['country'].str.endswith('(2001)')]
# Find the country with the highest percentage of orphans
max_orphans_country = df_2001.loc[df_2001['orphans as % of all children'].idxmax(), 'country']
print(f"Final Answer: {max_orphans_country}")