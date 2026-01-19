import pandas as pd

df = pd.read_csv('table.csv')

# Convert area and HDI to numeric, coercing errors to NaN if needed
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'], errors='coerce')
df['human development index (2000)'] = pd.to_numeric(df['human development index (2000)'], errors='coerce')

# Filter rows where area > 700 and HDI > 0.7
filtered_df = df[(df['area (km 2 )'] > 700) & (df['human development index (2000)'] > 0.7)]

# Calculate average population density of filtered rows
if not filtered_df.empty:
    avg_density = filtered_df['population density ( / km 2 )'].mean()
else:
    avg_density = 0

print(f"Final Answer: {avg_density:.2f}")