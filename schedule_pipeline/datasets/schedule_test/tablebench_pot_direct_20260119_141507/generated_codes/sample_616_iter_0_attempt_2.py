import pandas as pd

df = pd.read_csv('table.csv')
# Select top 5 shows (rank 1 to 5) and sum their hk viewers
total_viewership = df[df['rank'].isin(range(1, 6))]['hk viewers'].sum()
print(f"Final Answer: {total_viewership}")