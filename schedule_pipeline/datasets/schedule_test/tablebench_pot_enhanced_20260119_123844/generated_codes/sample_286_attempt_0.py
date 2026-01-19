import pandas as pd

df = pd.read_csv('table.csv')

# Extract the first number from "rating / share (1849)" and convert to float
df['rating_share_value'] = df['rating / share (1849)'].str.split('/').str[0].astype(float)

# Filter rows where rating/share >= 3.0 and rank (timeslot) is 1
filtered_df = df[(df['rating_share_value'] >= 3.0) & (df['rank (timeslot)'] == '1')]

# Sum the viewers (millions)
total_viewers = filtered_df['viewers (millions)'].sum()

print(f"Final Answer: {total_viewers:.2f}")