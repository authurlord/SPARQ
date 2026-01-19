import pandas as pd

df = pd.read_csv('table.csv')
# Extract and convert 'floors' to integer
df['floors'] = df['floors'].astype(int)
# Sort by height in descending order and take top 5
top_5_floors = df.nlargest(5, 'height ft / m')['floors']
# Calculate average
avg_floors = top_5_floors.mean()
print(f"Final Answer: {avg_floors:.1f}")