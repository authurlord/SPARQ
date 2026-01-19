import pandas as pd

df = pd.read_csv('table.csv')
# Convert height column to numeric by removing ' m' and converting to float
df['height'] = df['height'].str.replace(' m', '').astype(float)
# Get the top 5 buildings and increase their height by 5 meters
top_5_heights = df['height'].head(5) + 5
# Calculate the new average height
new_average_height = top_5_heights.mean()
print(f"Final Answer: {new_average_height:.1f}")