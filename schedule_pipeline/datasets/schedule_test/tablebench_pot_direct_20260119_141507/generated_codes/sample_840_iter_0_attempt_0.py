import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'passes through - district(s)' contains 'sirmour'
sirmour_roads = df[df['passes through - district (s)'].str.contains('sirmour', na=False)]
# Extract length values and compute difference between max and min
lengths = sirmour_roads['length (in km)'].astype(float)
difference = lengths.max() - lengths.min()
print(f"Final Answer: {difference:.2f}")