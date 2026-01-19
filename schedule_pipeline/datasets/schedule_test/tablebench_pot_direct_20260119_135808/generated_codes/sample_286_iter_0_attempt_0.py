import pandas as pd

df = pd.read_csv('table.csv')

# Filter episodes with rating/share (1849) >= 3.0 and rank (timeslot) == 1
filtered_df = df[(df['rating / share (1849)'] >= '3.0 / 9') | (df['rating / share (1849)'] >= '3.1 / 9') | (df['rating / share (1849)'] >= '3.2 / 8') | (df['rating / share (1849)'] >= '3.2 / 9') | (df['rating / share (1849)'] >= '3.4 / 10') | (df['rating / share (1849)'] >= '3.8 / 9') | (df['rating / share (1849)'] >= '3.9 / 10')]

# Since the column is string, we need to extract the first number (rating) and compare
# Extract the rating part from 'rating / share (1849)'
df['rating_only'] = df['rating / share (1849)'].str.split('/').str[0].astype(float)

# Now filter where rating >= 3.0 and rank (timeslot) == 1
filtered_df = df[(df['rating_only'] >= 3.0) & (df['rank (timeslot)'] == '1')]

# Sum the viewers (millions)
total_viewers = filtered_df['viewers (millions)'].astype(float).sum()

print(f"Final Answer: {total_viewers:.2f}")