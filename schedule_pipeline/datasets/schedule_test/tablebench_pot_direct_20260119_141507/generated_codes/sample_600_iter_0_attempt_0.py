import pandas as pd

df = pd.read_csv('table.csv')

# Convert height string to numeric (take first value before '/')
df['height_ft'] = df['height ft / m'].str.split('/').str[0].astype(float)

# Sort by height in descending order and get top 5
top_5 = df.sort_values(by='height_ft', ascending=False).head(5)

# Calculate average number of floors among top 5
avg_floors = top_5['floors'].mean()

print(f"Final Answer: {avg_floors:.1f}")