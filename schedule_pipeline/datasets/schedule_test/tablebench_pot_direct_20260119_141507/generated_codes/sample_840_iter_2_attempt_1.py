import pandas as pd

df = pd.read_csv('table.csv')
# Filter roads that pass through 'sirmour' district
sirmour_roads = df[df['passes through - district (s)'].str.contains('sirmour', case=False, na=False)]
# Extract lengths and compute max and min
lengths = sirmour_roads['length (in km)'].astype(float)
max_length = lengths.max()
min_length = lengths.min()
difference = max_length - min_length
print(f"Final Answer: {difference:.2f}")