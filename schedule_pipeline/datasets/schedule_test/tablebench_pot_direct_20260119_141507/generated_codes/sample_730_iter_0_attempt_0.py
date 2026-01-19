import pandas as pd

df = pd.read_csv('table.csv')
# Filter mountains in Greece (including country names with greece)
greece_mountains = df[df['country'].str.contains('greece', case=False, na=False)]
# Find the mountain with the highest elevation
highest_elevation_mountain = greece_mountains.loc[greece_mountains['elevation (m)'].idxmax(), 'peak']
print(f"Final Answer: {highest_elevation_mountain}")