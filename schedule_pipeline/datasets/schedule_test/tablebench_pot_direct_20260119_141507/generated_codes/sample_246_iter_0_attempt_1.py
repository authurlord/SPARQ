import pandas as pd

df = pd.read_csv('table.csv')
# Filter top 3 buildings (ranks 1, 2, 3) completed before 1980
filtered_df = df[(df['rank'].isin([1, 2, 3])) & (df['completed'].astype(int) < 1980)]
# Calculate average storeys
average_storeys = filtered_df['storeys'].mean()
print(f"Final Answer: {average_storeys}")