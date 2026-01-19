import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract height values, convert string to numeric (remove 'm' and convert)
heights = []
for height_str in df.iloc[:5]['height']:
    # Extract numeric value from string like "98 m (322ft)"
    height_val = float(height_str.split('m')[0])
    heights.append(height_val)

# Current average height of top 5 buildings
current_avg = sum(heights) / len(heights)

# New average after increasing each by 5 meters
new_avg = (current_avg + 5)

print(f"Final Answer: {new_avg:.1f}")