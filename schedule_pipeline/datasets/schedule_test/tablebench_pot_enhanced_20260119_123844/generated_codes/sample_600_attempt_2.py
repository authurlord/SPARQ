import pandas as pd

df = pd.read_csv('table.csv')
# Extract the height in meters and convert to float
df['height_m'] = df['height ft / m'].str.split('/').str[1].astype(float)
# Sort by height in descending order and get top 5
top_5 = df.sort_values('height_m', ascending=False).head(5)
# Calculate average floors for top 5
avg_floors = top_5['floors'].astype(int).mean()
print(f"Final Answer: {avg_floors:.1f}")