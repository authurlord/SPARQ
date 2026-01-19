import pandas as pd

df = pd.read_csv('table.csv')
# Convert height to numeric by removing ' m' and converting to float
df['height'] = df['height'].str.replace(' m \(.*\)', '', regex=True).astype(float)
# Get top 5 buildings by rank
top_5 = df.head(5)
# Increase each height by 5 meters
new_heights = top_5['height'] + 5
# Calculate new average
new_avg_height = new_heights.mean()
print(f"Final Answer: {new_avg_height:.1f}")