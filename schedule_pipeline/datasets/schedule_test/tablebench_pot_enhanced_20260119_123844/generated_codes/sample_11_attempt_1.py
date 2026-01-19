import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'height (m)' to numeric
df['height (m)'] = pd.to_numeric(df['height (m)'])
# Sort by height in descending order and take top 5
top_5_heights = df.nlargest(5, 'height (m)')['height (m)']
# Calculate average height
average_height = top_5_heights.mean()
print(f"Final Answer: {average_height:.1f}")