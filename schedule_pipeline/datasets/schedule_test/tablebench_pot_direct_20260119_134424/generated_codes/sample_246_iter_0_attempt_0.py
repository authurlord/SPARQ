import pandas as pd

df = pd.read_csv('table.csv')
# Filter buildings completed before 1980
completed_before_1980 = df[df['completed'] < '1980']
# Select top 3 by rank
top_3 = completed_before_1980.head(3)
# Calculate average storeys
avg_storeys = top_3['storeys'].mean()
print(f"Final Answer: {avg_storeys:.1f}")