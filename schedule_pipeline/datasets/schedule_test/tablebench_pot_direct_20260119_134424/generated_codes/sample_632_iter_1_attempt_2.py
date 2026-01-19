import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'height' column by extracting the numeric value before 'm'
df['height'] = df['height'].str.extract(r'(\d+)').astype(float)
# Get the top 5 buildings by rank
top_5_heights = df['height'].head(5)
# Increase each height by 5 meters
new_heights = top_5_heights + 5
# Calculate the new average
new_average = new_heights.mean()
print(f"Final Answer: {new_average:.1f}")