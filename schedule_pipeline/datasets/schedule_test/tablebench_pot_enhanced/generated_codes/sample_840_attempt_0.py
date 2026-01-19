import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where district is 'sirmour'
sirmour_roads = df[df['passes through - district (s'] == 'sirmour']
# Convert length to float
sirmour_roads['length (in km)'] = pd.to_numeric(sirmour_roads['length (in km)'])
# Find max and min length
max_length = sirmour_roads['length (in km)'].max()
min_length = sirmour_roads['length (in km)'].min()
# Calculate difference
difference = max_length - min_length
print(f"Final Answer: {difference:.1f}")