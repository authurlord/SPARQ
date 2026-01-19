import pandas as pd
import re

df = pd.read_csv('table.csv')

# Filter rows for year 1944
year_1944 = df[df['Year'] == '1944']

# Extract numeric part from 'US Chart position' (remove text like '(R&B)')
def extract_position(pos):
    # Use regex to extract only digits
    match = re.search(r'\d+', pos)
    return int(match.group()) if match else None

# Apply extraction and filter out invalid entries
positions = year_1944['US Chart position'].apply(extract_position)
# Drop NaN values (if any)
positions = positions.dropna()

# Calculate average
average_position = positions.mean()
print(f"Final Answer: {average_position:.1f}")