import pandas as pd

df = pd.read_csv('table.csv')
# Filter mountains with elevation >= 3000 meters
high_elevation_mountains = df[df['elevation (m)'].astype(int) >= 3000]
count = len(high_elevation_mountains)
print(f"Final Answer: {count}")