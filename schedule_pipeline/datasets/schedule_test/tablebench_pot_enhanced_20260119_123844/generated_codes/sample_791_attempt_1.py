import pandas as pd

df = pd.read_csv('table.csv')
# Select top 10 airports based on rank
top_10 = df.head(10)
# Calculate median of aircraft movements
median_movements = top_10['aircraft movements'].median()
print(f"Final Answer: {median_movements}")