import pandas as pd

df = pd.read_csv('table.csv')
# Filter roads that pass through 'sirmour'
sirmour_roads = df[df['passes through - district (s)'].str.contains('sirmour', case=False, na=False)]
# Extract lengths and compute difference between max and min
lengths = sirmour_roads['length (in km)'].astype(float)
difference = lengths.max() - lengths.min()
print(f"Final Answer: {difference:.2f}")