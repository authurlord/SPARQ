import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter rows from 1942 to 1946 inclusive
filtered_df = df[(df['Year'].between(1942, 1946))]

# Extract numeric US Chart position (remove any text like "(R&B)")
def extract_position(pos):
    # Remove any non-numeric characters except digits and decimal points
    import re
    match = re.search(r'(\d+)', pos)
    return float(match.group(1)) if match else np.nan

# Apply the function to the 'US Chart position' column
positions = filtered_df['US Chart position'].apply(extract_position)

# Calculate the mean of valid positions
average_position = positions.mean()

print(f"Final Answer: {average_position:.1f}")