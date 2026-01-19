import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'], errors='coerce')
df['human development index (2000)'] = pd.to_numeric(df['human development index (2000)'], errors='coerce')

# Filter municipalities with area > 700 and HDI > 0.7
filtered_df = df[(df['area (km 2 )'] > 700) & (df['human development index (2000)'] > 0.7)]

# Calculate average population density of filtered municipalities
average_density = filtered_df['population density ( / km 2 )'].mean()

print(f"Final Answer: {average_density:.2f}")