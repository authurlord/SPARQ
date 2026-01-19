import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert 'bp comp 1 (˚c)' to numeric, handling errors by converting '-' to NaN
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')

# Find max and min values
max_bp1 = df['bp comp 1 (˚c)'].max()
min_bp1 = df['bp comp 1 (˚c)'].min()

# Calculate the difference
difference = max_bp1 - min_bp1
print(f"Final Answer: {difference:.1f}")