import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bp comp 1 (˚c)' to numeric, handling any potential non-numeric entries
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')
# Find max and min values
max_bp = df['bp comp 1 (˚c)'].max()
min_bp = df['bp comp 1 (˚c)'].min()
# Calculate difference
difference = max_bp - min_bp
print(f"Final Answer: {difference:.1f}")