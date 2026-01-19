import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the row for Copper (mg) - it's at index 26
copper_row = df.iloc[26]

# Get the staple food names (columns from 'Maize / Corn' onwards)
staples = df.columns[1:]  # Skip the first column 'STAPLE:'
copper_values = copper_row[1:]  # Skip the first value (label)

# Find the staple with the highest Copper (mg)
max_copper_staple = staples[copper_values.idxmax()]

print(f"Final Answer: {max_copper_staple}")