import pandas as pd

df = pd.read_csv('table.csv')
# Filter data from May 1944 to August 1944
filtered_df = df[df['Month'].isin(['May 1944', 'June 1944', 'July 1944', 'August 1944'])]
# Convert columns to integers
filtered_df[['M36', 'M36B1', 'M36B2']] = filtered_df[['M36', 'M36B1', 'M36B2']].astype(int)
# Calculate total medals
total_medals = filtered_df['M36'].sum() + filtered_df['M36B1'].sum() + filtered_df['M36B2'].sum()
print(f"Final Answer: {total_medals}")