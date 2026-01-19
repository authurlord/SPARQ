import pandas as pd

df = pd.read_csv('table.csv')
# Convert height from string to numeric (remove 'm' and convert to float)
df['height'] = df['height'].str.replace(' m', '').str.replace('ft', '').astype(float)
# Get the top 5 buildings by rank
top_5_heights = df['height'].head(5)
# Increase each height by 5 meters
new_heights = top_5_heights + 5
# Calculate new average
new_average = new_heights.mean()
print(f"Final Answer: {new_average:.1f}")