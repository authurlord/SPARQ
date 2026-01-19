import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'obama%' to float for comparison
df['obama%'] = df['obama%'].str.rstrip('%').astype(float)
# Filter counties where Obama received more than 60% of the votes
filtered_df = df[df['obama%'] > 60]
# Sum the total votes in these counties
total_votes = filtered_df['total'].sum()
print(f"Final Answer: {total_votes}")