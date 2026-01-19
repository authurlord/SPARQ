import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where country contains 'greece' (case-insensitive)
greece_mountains = df[df['country'].str.contains('greece', case=False, na=False)]
# Convert elevation to integer and find the mountain with max elevation
max_elevation_mountain = greece_mountains.loc[greece_mountains['elevation (m)'].astype(int).idxmax(), 'peak']
print(f"Final Answer: {max_elevation_mountain}")