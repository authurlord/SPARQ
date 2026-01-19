import pandas as pd

df = pd.read_csv('table.csv')
# Extract numeric part from 'height' column (before 'm')
df['height_m'] = df['height'].str.extract(r'(\d+)').astype(float)
# Get top 5 buildings
top_5_heights = df['height_m'].head(5)
# Increase each by 5 meters
new_heights = top_5_heights + 5
# Calculate new average
new_average = new_heights.mean()
print(f"Final Answer: {new_average:.1f}")