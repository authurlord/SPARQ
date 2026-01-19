import pandas as pd

df = pd.read_csv('table.csv')
# Convert viewers to numeric (they are strings with commas)
df['viewers'] = df['viewers'].str.replace(',', '').astype(int)

# Calculate average viewership and average BBC Three weekly ranking
avg_viewers = df['viewers'].mean()
avg_ranking = df['bbc three weekly ranking'].mean()

print(f"Final Answer: {avg_viewers:.0f}, {avg_ranking:.0f}")