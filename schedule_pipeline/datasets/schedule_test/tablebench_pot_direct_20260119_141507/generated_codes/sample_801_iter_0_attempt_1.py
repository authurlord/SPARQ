import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'diameter (km)' to numeric, handling any potential parsing issues
df['diameter (km)'] = pd.to_numeric(df['diameter (km)'], errors='coerce')
# Drop any rows with NaN due to invalid entries
df = df.dropna(subset=['diameter (km)'])
# Calculate the median diameter
median_diameter = df['diameter (km)'].median()
print(f"Final Answer: {median_diameter:.1f}")