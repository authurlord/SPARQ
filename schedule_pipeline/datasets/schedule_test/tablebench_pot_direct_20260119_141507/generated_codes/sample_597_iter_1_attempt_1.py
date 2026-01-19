import pandas as pd

df = pd.read_csv('table.csv')
# Sort by 'no in series' to get the first 5 episodes
df_sorted = df.sort_values(by='no in series').head(5)
# Calculate the average viewers for these 5 episodes
average_viewers = df_sorted['viewers (millions)'].mean()
print(f"Final Answer: {average_viewers:.2f}")