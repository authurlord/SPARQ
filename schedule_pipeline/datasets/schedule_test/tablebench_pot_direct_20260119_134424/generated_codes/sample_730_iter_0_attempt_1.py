import pandas as pd

df = pd.read_csv('table.csv')
# Filter mountains in Greece (case-insensitive)
greece_mountains = df[df['country'].str.contains('greece', case=False, na=False)]
# Convert elevation to integer and find the max
max_elevation_row = greece_mountains.loc[greece_mountains['elevation (m)'].astype(int).idxmax()]
print(f"Final Answer: {max_elevation_row['peak']}")