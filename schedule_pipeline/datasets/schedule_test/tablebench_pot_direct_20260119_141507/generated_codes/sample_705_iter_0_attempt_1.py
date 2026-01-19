import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Drop the first row (header row) as it contains metadata
data_rows = df.iloc[1:]

# Extract the 'Copper (mg)' column and the corresponding staple food names
copper_values = data_rows['Copper (mg)']
staple_foods = data_rows.index

# Find the index of the maximum copper value
max_copper_index = copper_values.idxmax()

# Get the corresponding staple food name
max_copper_staple = data_rows.loc[max_copper_index, 'STAPLE:']

# Since the column 'STAPLE:' is not directly aligned with the values, we need to map correctly.
# Actually, the 'STAPLE:' column is the first column and contains the food names.
# So we should use the index of the data row to get the food name.

# Correct approach: extract the food name from the first column
food_names = data_rows['STAPLE:'].values
copper_values = data_rows['Copper (mg)'].values

# Find the food with maximum copper
max_copper_idx = copper_values.argmax()
highest_copper_food = food_names[max_copper_idx]

print(f"Final Answer: {highest_copper_food}")