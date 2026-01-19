import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rating / share (1849)' to float for comparison
df['rating / share (1849)'] = df['rating / share (1849)'].str.split('/').str[0].astype(float)
# Filter rows where rating/share is >= 3.0 and rank (timeslot) is 1
filtered_df = df[(df['rating / share (1849)'] >= 3.0) & (df['rank (timeslot)'] == '1')]
# Sum the viewers (millions) for these episodes
total_viewers = filtered_df['viewers (millions)'].sum()
print(f"Final Answer: {total_viewers:.2f}")