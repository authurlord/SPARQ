import pandas as pd

df = pd.read_csv('table.csv')

# Convert height string to numeric (take the first number)
df['height_ft'] = df['height ft / m'].str.split('/').str[0].astype(float)

# Sort by height in descending order and take top 5
top_5 = df.sort_values(by='height_ft', ascending=False).head(5)

# Calculate average floors among the top 5
avg_floors = top_5['floors'].mean()
print(f"Final Answer: {avg_floors:.1f}")