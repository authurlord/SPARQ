import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Parse height column to extract feet value
df['height_ft'] = df['height ft / m'].str.split('/').str[0].astype(float)

# Sort by height (descending) and get top 5
top_5 = df.sort_values(by='height_ft', ascending=False).head(5)

# Calculate average floors among these top 5
avg_floors = top_5['floors'].mean()

print(f"Final Answer: {avg_floors:.1f}")