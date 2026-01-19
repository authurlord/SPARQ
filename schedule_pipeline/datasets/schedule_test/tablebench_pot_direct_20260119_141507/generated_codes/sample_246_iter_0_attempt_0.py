import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 3 buildings by rank (ranks 1, 2, 3)
top_3 = df[df['rank'].isin([1, 2, 3])]
# Filter only those completed before 1980
before_1980 = top_3[top_3['completed'].astype(int) < 1980]
# Calculate average storeys
avg_storeys = before_1980['storeys'].mean()
print(f"Final Answer: {avg_storeys}")