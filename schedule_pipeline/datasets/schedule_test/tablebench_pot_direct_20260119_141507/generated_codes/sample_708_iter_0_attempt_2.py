import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter rows where the country entry has '2001' in it
filtered_df = df[df['country'].str.contains('2001', na=False)]

# Extract the country and the 'orphans as % of all children' value
filtered_df['orphans as % of all children'] = filtered_df['orphans as % of all children'].astype(float)

# Find the country with the highest percentage
max_orphan_percentage = filtered_df.loc[filtered_df['orphans as % of all children'].idxmax()]
highest_country = max_orphan_percentage['country'].split(' (2001)')[0]

print(f"Final Answer: {highest_country}")