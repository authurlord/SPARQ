import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bp comp 1 (˚c)' to numeric, coercing errors to NaN
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')
# Drop rows with NaN values if any
df.dropna(subset=['bp comp 1 (˚c)'], inplace=True)
# Find max and min boiling points
max_bp = df['bp comp 1 (˚c)'].max()
min_bp = df['bp comp 1 (˚c)'].min()
# Calculate difference
difference = max_bp - min_bp
print(f"Final Answer: {difference}")