import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'floors' column and convert to integer
df['floors'] = df['floors'].astype(int)
# Select the top 5 tallest buildings (first 5 rows)
top_5_floors = df['floors'].head(5)
# Calculate the average
avg_floors = top_5_floors.mean()
print(f"Final Answer: {avg_floors:.1f}")