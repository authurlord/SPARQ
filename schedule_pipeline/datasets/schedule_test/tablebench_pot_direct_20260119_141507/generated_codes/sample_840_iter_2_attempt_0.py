import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'passes through - district(s)' contains 'sirmour'
sirmour_roads = df[df['passes through - district (s)'].str.contains('sirmour', na=False)]
# Extract lengths and convert to float
lengths = sirmour_roads['length (in km)'].astype(float)
# Calculate difference between max and min lengths
length_difference = lengths.max() - lengths.min()
print(f"Final Answer: {length_difference:.2f}")