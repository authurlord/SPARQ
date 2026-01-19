import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'area (km 2 )' and 'human development index (2000)' to float for comparison
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'], errors='coerce')
df['human development index (2000)'] = pd.to_numeric(df['human development index (2000)'], errors='coerce')
df['population density ( / km 2 )'] = pd.to_numeric(df['population density ( / km 2 )'], errors='coerce')

# Filter rows based on conditions
filtered_df = df[(df['area (km 2 )'] > 700) & (df['human development index (2000)'] > 0.7)]

# Calculate average population density
avg_density = filtered_df['population density ( / km 2 )'].mean()

print(f"Final Answer: {avg_density:.2f}")