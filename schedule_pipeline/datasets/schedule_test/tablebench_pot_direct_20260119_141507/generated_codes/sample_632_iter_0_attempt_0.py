import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract height values, convert string like '98 m (322ft)' to numeric
df['height'] = df['height'].str.extract(r'(\d+)').astype(float)

# Sort by height in descending order and take top 5
top_5 = df.nlargest(5, 'height')

# Current average height of top 5
current_avg = top_5['height'].mean()

# Increase each height by 5 meters and compute new average
new_heights = top_5['height'] + 5
new_avg = new_heights.mean()

print(f"Final Answer: {new_avg:.1f}")