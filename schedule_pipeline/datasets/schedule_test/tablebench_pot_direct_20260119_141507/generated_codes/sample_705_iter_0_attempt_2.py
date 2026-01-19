import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Select the row for Copper (mg) - it's at index 15 (0-based)
copper_row = df.iloc[15]

# Drop the first column 'STAPLE:' since it's not a food item
# Extract only the food names (columns from index 1 onwards)
food_names = df.columns[1:]  # All columns except the first

# Get the copper values for each food
copper_values = copper_row[1:]

# Find the food with the highest copper value
max_copper_index = copper_values.idxmax()
highest_copper_food = max_copper_index

print(f"Final Answer: {highest_copper_food}")