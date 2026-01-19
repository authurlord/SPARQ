import pandas as pd

df = pd.read_csv('table.csv')
# Extract floors and convert to integer
df['floors'] = df['floors'].astype(int)
# Take top 5 tallest buildings (first 5 rows)
top_5_floors = df['floors'].head(5)
# Calculate average
avg_floors = top_5_floors.mean()
print(f"Final Answer: {avg_floors:.1f}")