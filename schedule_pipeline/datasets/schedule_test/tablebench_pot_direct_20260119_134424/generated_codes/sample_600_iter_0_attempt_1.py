import pandas as pd

df = pd.read_csv('table.csv')
# Extract the numeric part of 'height ft / m' column (in meters)
df['height_m'] = df['height ft / m'].str.split('/').str[1].astype(float)
# Sort by height in descending order and take top 5
top_5 = df.nlargest(5, 'height_m')
# Calculate average floors for top 5
avg_floors = top_5['floors'].mean()
print(f"Final Answer: {avg_floors:.1f}")