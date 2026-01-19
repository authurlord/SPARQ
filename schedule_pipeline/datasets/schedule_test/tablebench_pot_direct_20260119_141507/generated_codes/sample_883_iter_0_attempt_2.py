import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where the year is 2001
df_2001 = df[df['country'].str.contains(r'2001', na=False)]

# Extract the percentage of AIDS-related orphans as a percentage of total orphans
aids_orphan_percentage = df_2001['aids orphans as % of orphans'].astype(float)

# Find the country with the highest percentage
max_percentage = aids_orphan_percentage.max()
highest_country = df_2001.loc[aids_orphan_percentage.idxmax(), 'country']

# Get Uganda's percentage in 2001
uganda_2001 = df[df['country'] == 'uganda (2001)']['aids orphans as % of orphans'].astype(float).iloc[0]

print(f"Final Answer: {highest_country}, {uganda_2001}")