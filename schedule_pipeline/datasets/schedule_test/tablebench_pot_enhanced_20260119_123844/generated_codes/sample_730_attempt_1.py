import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where country contains 'greece' (case-insensitive)
greece_mountains = df[df['country'].str.contains('greece', case=False, na=False)]
# Convert elevation to integer for comparison
greece_mountains['elevation (m)'] = pd.to_numeric(greece_mountains['elevation (m)'])
# Find the mountain with the highest elevation
highest_elevation_mountain = greece_mountains.loc[greece_mountains['elevation (m)'].idxmax(), 'peak']
print(f"Final Answer: {highest_elevation_mountain}")