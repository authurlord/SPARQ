import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for May 1944 to August 1944
filtered_df = df[(df['Month'] >= 'May 1944') & (df['Month'] <= 'August 1944')]
# Convert columns to integers and calculate total medals
total_medals = filtered_df[['M36', 'M36B1', 'M36B2']].astype(int).sum().sum()
print(f"Final Answer: {total_medals}")