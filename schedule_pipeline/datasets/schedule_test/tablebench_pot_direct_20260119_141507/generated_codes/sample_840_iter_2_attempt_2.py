import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'passes through - district (s' contains 'sirmour'
sirmour_roads = df[df['passes through - district (s'] == 'sirmour']
# Extract lengths and convert to float
lengths = sirmour_roads['length (in km)'].astype(float)
# Find max and min lengths
max_length = lengths.max()
min_length = lengths.min()
difference = max_length - min_length
print(f"Final Answer: {difference:.2f}")