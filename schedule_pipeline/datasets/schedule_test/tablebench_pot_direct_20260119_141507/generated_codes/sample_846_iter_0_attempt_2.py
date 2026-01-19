import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bp comp 1 (˚c)' column to numeric, handling strings like '- 0.5'
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'].str.replace(' - ', ' -'), errors='coerce')

# Find max and min boiling points
max_bp = df['bp comp 1 (˚c)'].max()
min_bp = df['bp comp 1 (˚c)'].min()

difference = max_bp - min_bp
print(f"Final Answer: {difference:.1f}")