import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row which is a header row (STAPLE:)
# The actual data starts from row index 1
data_rows = df.iloc[1:]

# Select the 'Copper (mg)' column and the corresponding staple food names
copper_values = data_rows['Copper (mg)'].dropna()
staple_foods = data_rows.iloc[:, 0]  # First column is the staple food name

# Find the staple food with the maximum copper value
max_copper_index = copper_values.idxmax()
highest_copper_staple = staple_foods[max_copper_index]

print(f"Final Answer: {highest_copper_staple}")