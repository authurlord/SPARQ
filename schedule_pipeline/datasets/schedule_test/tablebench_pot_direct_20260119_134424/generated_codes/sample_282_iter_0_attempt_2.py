import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to appropriate types
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'])
df['human development index (2000)'] = pd.to_numeric(df['human development index (2000)'])
df['population density ( / km 2 )'] = pd.to_numeric(df['population density ( / km 2 )'])

# Filter municipalities with area > 700 km² and HDI > 0.7
filtered_df = df[(df['area (km 2 )'] > 700) & (df['human development index (2000)'] > 0.7)]

# Calculate average population density
avg_density = filtered_df['population density ( / km 2 )'].mean()

print(f"Final Answer: {avg_density:.1f}")