import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes with rating >= 6.0
filtered_df = df[df['rating'] >= '6.0']
# Convert 'viewers (millions)' to float and calculate mean
average_viewers = filtered_df['viewers (millions)'].astype(float).mean()
print(f"Final Answer: {average_viewers:.2f}")