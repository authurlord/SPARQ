import pandas as pd

df = pd.read_csv('table.csv')
# Sort by passengers in descending order and take top 10
top_10 = df.sort_values(by='passengers', ascending=False).head(10)
# Extract aircraft movements for top 10
movements = top_10['aircraft movements'].astype(float)
# Calculate median
median_movements = movements.median()
print(f"Final Answer: {median_movements}")