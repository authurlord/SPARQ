import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'rating / share (1849)' to numeric by splitting on '/'
df['rating_share'] = df['rating / share (1849)'].str.split('/').str[0].astype(float)

# Filter episodes with rating/share >= 3.0 and rank (timeslot) = 1
filtered_df = df[(df['rating_share'] >= 3.0) & (df['rank (timeslot)'] == '1')]

# Sum the viewers (millions)
total_viewers = filtered_df['viewers (millions)'].sum()

print(f"Final Answer: {total_viewers:.2f}")