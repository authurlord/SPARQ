import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for months from May 1944 to August 1944
filtered_df = df[(df['Month'] >= 'May 1944') & (df['Month'] <= 'August 1944')]

# Calculate the total medals (sum of M36 + M36B1 + M36B2) for the filtered rows
total_medals = filtered_df['M36'].sum() + filtered_df['M36B1'].sum() + filtered_df['M36B2'].sum()
print(f"Final Answer: {total_medals}")