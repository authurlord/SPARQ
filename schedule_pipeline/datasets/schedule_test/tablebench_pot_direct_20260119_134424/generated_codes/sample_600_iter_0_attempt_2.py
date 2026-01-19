import pandas as pd

df = pd.read_csv('table.csv')
# Extract and convert 'floors' to integer
df['floors'] = df['floors'].astype(int)
# Sort by height (convert height to numeric for sorting)
df['height_ft'] = df['height ft / m'].str.split('/').str[0].astype(float)
# Get top 5 tallest buildings
top_5 = df.nlargest(5, 'height_ft')
# Calculate average floors
avg_floors = top_5['floors'].mean()
print(f"Final Answer: {avg_floors:.1f}")