import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'Office started' and 'Office ended' to integers, if possible
df['Office started'] = pd.to_numeric(df['Office started'], errors='coerce')
df['Office ended'] = pd.to_numeric(df['Office ended'], errors='coerce')

# Filter out rows where either start or end is missing (invalid tenure)
valid_rows = df.dropna(subset=['Office started', 'Office ended'])

# Calculate tenure in years
valid_rows['tenure'] = valid_rows['Office ended'] - valid_rows['Office started']

# Find the bishop with the longest tenure
longest_tenure_bishop = valid_rows.loc[valid_rows['tenure'].idxmax(), 'Name']
max_tenure = valid_rows['tenure'].max()

# Calculate average tenure
avg_tenure = valid_rows['tenure'].mean()

# Difference between longest and average tenure
difference = max_tenure - avg_tenure

print(f"Final Answer: {longest_tenure_bishop}, {difference:.1f}")