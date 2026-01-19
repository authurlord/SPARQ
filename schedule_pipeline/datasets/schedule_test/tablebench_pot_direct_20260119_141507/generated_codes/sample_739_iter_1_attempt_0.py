import pandas as pd

df = pd.read_csv('table.csv')
# Convert viewers to numeric
df['viewers'] = pd.to_numeric(df['viewers'], errors='coerce')

# Calculate mean viewership and mean BBC Three weekly ranking
mean_viewers = df['viewers'].mean()
mean_ranking = df['bbc three weekly ranking'].mean()

print(f"Final Answer: {mean_viewers:.0f}, {mean_ranking:.0f}")