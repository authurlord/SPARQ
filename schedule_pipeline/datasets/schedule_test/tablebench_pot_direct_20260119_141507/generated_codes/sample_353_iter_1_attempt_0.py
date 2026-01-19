import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'per capita income' to integer
df['per capita income'] = df['per capita income'].str.replace(',', '').astype(int)
# Filter counties with per capita income between 18000 and 27000 (inclusive)
counties_in_range = df[(df['per capita income'] >= 18000) & (df['per capita income'] <= 27000)]
# Count the number of such counties
result = len(counties_in_range)
print(f"Final Answer: {result}")