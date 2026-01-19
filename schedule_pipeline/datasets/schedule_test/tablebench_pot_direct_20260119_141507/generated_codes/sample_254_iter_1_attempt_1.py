import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation (m) to numeric, coercing errors to NaN
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')

# Filter districts in Cusco region with elevation > 4700
filtered_df = df[(df['region'] == 'cusco') & (df['elevation (m)'] > 4700)]

# Calculate average elevation
if not filtered_df.empty:
    avg_elevation = filtered_df['elevation (m)'].mean()
else:
    avg_elevation = 0

print(f"Final Answer: {avg_elevation:.1f}")