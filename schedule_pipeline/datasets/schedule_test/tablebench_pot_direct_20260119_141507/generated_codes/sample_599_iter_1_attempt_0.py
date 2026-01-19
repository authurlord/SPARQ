import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter rows for year 1944
filtered_df = df[df['Year'] == '1944']

# Extract numeric US Chart position by removing text in parentheses
def extract_position(pos):
    # Remove any text after a space or in parentheses
    if '(' in pos:
        return int(pos.split('(')[0].strip())
    else:
        return int(pos)

# Apply extraction and compute mean
positions = filtered_df['US Chart position'].apply(extract_position)
average_position = positions.mean()

print(f"Final Answer: {average_position:.1f}")