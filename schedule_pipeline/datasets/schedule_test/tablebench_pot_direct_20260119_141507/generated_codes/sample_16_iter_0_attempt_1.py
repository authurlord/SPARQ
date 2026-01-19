import pandas as pd
import re

df = pd.read_csv('table.csv')
# Extract numeric US Chart position from the 'US Chart position' column
def extract_position(pos):
    match = re.search(r'(\d+)', pos)
    return int(match.group(1)) if match else None

# Apply the function to get clean positions
positions = df['US Chart position'].apply(extract_position)
# Filter out any invalid entries (if any)
positions = positions.dropna()
# Calculate the mean
avg_position = positions.mean()
print(f"Final Answer: {avg_position:.1f}")