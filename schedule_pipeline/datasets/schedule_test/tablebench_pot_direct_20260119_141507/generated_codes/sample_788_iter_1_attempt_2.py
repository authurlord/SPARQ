import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'elevation (m)' to numeric, coercing errors to NaN if any, then drop invalid entries
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
# Calculate median of valid elevation values
median_elevation = df['elevation (m)'].median()
print(f"Final Answer: {median_elevation}")