import pandas as pd

df = pd.read_csv('table.csv')
# Filter municipalities with area > 700 km² and HDI > 0.7
filtered_df = df[(df['area (km 2 )'] > 700) & (df['human development index (2000)'] > 0.7)]
# Calculate the average population density of the filtered rows
avg_density = filtered_df['population density ( / km 2 )'].mean()
print(f"Final Answer: {avg_density:.2f}")