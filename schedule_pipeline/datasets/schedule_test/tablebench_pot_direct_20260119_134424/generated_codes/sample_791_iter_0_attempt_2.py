import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'aircraft movements' to integer
df['aircraft movements'] = pd.to_numeric(df['aircraft movements'])
# Get the top 10 airports by passengers
top_10 = df.head(10)
# Calculate median of aircraft movements for top 10
median_movements = top_10['aircraft movements'].median()
print(f"Final Answer: {median_movements}")