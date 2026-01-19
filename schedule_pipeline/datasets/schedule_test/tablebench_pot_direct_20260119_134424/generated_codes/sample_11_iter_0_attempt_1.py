import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'height (m)' to numeric
df['height (m)'] = pd.to_numeric(df['height (m)'])
# Sort by height in descending order and get top 5
top_5_heights = df.sort_values('height (m)', ascending=False).head(5)['height (m)']
# Calculate average
avg_height = top_5_heights.mean()
print(f"Final Answer: {avg_height:.1f}")