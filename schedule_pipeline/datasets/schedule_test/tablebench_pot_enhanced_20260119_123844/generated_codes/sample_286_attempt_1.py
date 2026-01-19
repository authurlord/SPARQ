import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where rating/share (1849) >= 3.0 and rank (timeslot) == 1
filtered_df = df[(df['rating / share (1849)'] >= '3.0 / 9') | (df['rating / share (1849)'] >= '3.1 / 9') | (df['rating / share (1849)'] >= '3.2 / 8') | (df['rating / share (1849)'] >= '3.4 / 10') | (df['rating / share (1849)'] >= '3.8 / 9') | (df['rating / share (1849)'] >= '3.9 / 10')]
# Since the column is string, we need to compare properly by extracting numeric part
# We'll filter based on the numeric value of the first part of 'rating / share (1849)'
filtered_df = df[df['rating / share (1849)'].str.split('/').str[0].astype(float) >= 3.0]
filtered_df = filtered_df[filtered_df['rank (timeslot)'] == '1']
# Convert viewers (millions) to float and sum
total_viewers = filtered_df['viewers (millions)'].astype(float).sum()
print(f"Final Answer: {total_viewers:.2f}")