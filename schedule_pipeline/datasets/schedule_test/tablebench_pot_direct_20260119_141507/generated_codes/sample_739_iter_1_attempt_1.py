import pandas as pd

df = pd.read_csv('table.csv')
# Calculate average viewership and BBC Three weekly ranking
avg_viewership = df['viewers'].mean()
avg_ranking = df['bbc three weekly ranking'].mean()
print(f"Final Answer: {avg_viewership:.0f}, {avg_ranking:.0f}")