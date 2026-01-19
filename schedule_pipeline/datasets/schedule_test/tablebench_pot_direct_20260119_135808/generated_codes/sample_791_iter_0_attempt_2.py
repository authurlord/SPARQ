import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'aircraft movements' to integer
df['aircraft movements'] = pd.to_numeric(df['aircraft movements'])
# Get top 10 airports (first 10 rows by rank)
top_10 = df.head(10)['aircraft movements']
# Calculate median
median_movements = top_10.median()
print(f"Final Answer: {median_movements}")