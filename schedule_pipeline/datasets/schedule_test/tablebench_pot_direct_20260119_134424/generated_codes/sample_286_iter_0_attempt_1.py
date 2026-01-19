import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where rating/share (1849) >= 3.0 and rank (timeslot) == 1
filtered_df = df[(df['rating / share (1849)'] >= '3.0 / 9') | (df['rating / share (1849)'] >= '3.0 / 10')]
# Convert viewers to float and sum
total_viewers = filtered_df['viewers (millions)'].astype(float).sum()
print(f"Final Answer: {total_viewers:.2f}")