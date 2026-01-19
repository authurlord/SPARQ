import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'passes through - district (s' contains 'sirmour'
sirmour_roads = df[df['passes through - district (s)'].str.contains('sirmour', case=False, na=False)]
# Extract the length column and convert to float
lengths = sirmour_roads['length (in km)'].astype(float)
# Calculate the difference between the longest and shortest road
difference = lengths.max() - lengths.min()
print(f"Final Answer: {difference:.2f}")