import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where organization is 'transparency international'
ti_rows = df[df['organization'] == 'transparency international']
# Convert rank to numeric and compute mean
mean_rank = ti_rows['rank'].astype(float).mean()
print(f"Final Answer: {mean_rank:.1f}")