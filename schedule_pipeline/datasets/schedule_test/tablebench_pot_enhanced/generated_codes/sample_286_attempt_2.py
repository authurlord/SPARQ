import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where rating/share (1849) is 3.0 or higher and rank (timeslot) is 1
filtered_df = df[(df['rating / share (1849)'] >= '3.0 / 9') | (df['rating / share (1849)'] >= '3.1 / 9') | (df['rating / share (1849)'] >= '3.2 / 8') | (df['rating / share (1849)'] >= '3.4 / 10') | (df['rating / share (1849)'] >= '3.8 / 9') | (df['rating / share (1849)'] >= '3.9 / 10')] 
# Convert 'viewers (millions)' to float and sum
total_viewers = filtered_df['viewers (millions)'].astype(float).sum()
print(f"Final Answer: {total_viewers:.2f}")