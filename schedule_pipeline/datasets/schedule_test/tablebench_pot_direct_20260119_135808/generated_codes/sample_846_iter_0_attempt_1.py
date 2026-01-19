import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'bp comp 1 (˚c)' to numeric, handling negative signs and spaces
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'].str.strip().str.replace('−', '-'), errors='coerce')
# Find max and min boiling points
max_bp = df['bp comp 1 (˚c)'].max()
min_bp = df['bp comp 1 (˚c)'].min()
# Calculate difference
difference = max_bp - min_bp
print(f"Final Answer: {difference:.1f}")