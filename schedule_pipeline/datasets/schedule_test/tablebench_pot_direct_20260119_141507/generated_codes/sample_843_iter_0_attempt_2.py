import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Drop rows where 'Office ended' is missing or invalid (e.g., "Incumbent")
df_clean = df.dropna(subset=['Office ended'])

# Convert Office started and Office ended to integers
df_clean['Office started'] = df_clean['Office started'].astype(int)
df_clean['Office ended'] = df_clean['Office ended'].astype(int)

# Calculate tenure in years
df_clean['tenure'] = df_clean['Office ended'] - df_clean['Office started']

# Find the bishop with the longest tenure
longest_tenure_bishop = df_clean.loc[df_clean['tenure'].idxmax(), 'Name']
max_tenure = df_clean['tenure'].max()

# Calculate average tenure
avg_tenure = df_clean['tenure'].mean()

# Difference between longest and average tenure
difference = max_tenure - avg_tenure

print(f"Final Answer: {longest_tenure_bishop}, {difference:.1f}")