import pandas as pd

df = pd.read_csv('table.csv')

# Extract the 'foreign' column
foreign_spoken = df['foreign']

# Find the maximum and minimum values and their corresponding municipalities
max_value = foreign_spoken.max()
min_value = foreign_spoken.min()

# Get the municipality names where max and min occur
max_municipality = df.index[foreign_spoken.idxmax()]
min_municipality = df.index[foreign_spoken.idxmin()]

# Calculate the difference
difference = max_value - min_value

print(f"Final Answer: {max_municipality}, {min_municipality}, {difference:.1f}")