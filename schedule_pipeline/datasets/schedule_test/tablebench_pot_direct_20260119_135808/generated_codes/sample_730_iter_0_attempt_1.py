import pandas as pd

df = pd.read_csv('table.csv')
# Filter mountains in Greece (case-insensitive)
greece_mountains = df[df['country'].str.contains('greece', case=False, na=False)]
# Convert elevation to integer and find the max
max_elevation_index = greece_mountains['elevation (m)'].astype(int).idxmax()
# Get the peak name with highest elevation
highest_peak = greece_mountains.loc[max_elevation_index, 'peak']
print(f"Final Answer: {highest_peak}")