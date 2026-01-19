import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract height values (convert string like "98 m (322ft)" to numeric)
heights = df['height'].str.extract(r'(\d+)').astype(float)

# Sort by height in descending order and get top 5
top_5_heights = heights.nlargest(5)

# Current average of top 5
current_avg = top_5_heights.mean()

# New average after increasing by 5 meters
new_avg = current_avg + 5

print(f"Final Answer: {new_avg:.1f}")