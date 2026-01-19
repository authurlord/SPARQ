import pandas as pd

df = pd.read_csv('table.csv')

# Find the row corresponding to 'foreign' language
foreign_row = df[df['language'] == 'foreign']

# Extract the values for foreign language across municipalities
foreign_values = foreign_row.iloc[0][1:]  # Drop the 'language' column

# Get the index of max and min values
max_idx = foreign_values.idxmax()
min_idx = foreign_values.idxmin()

# Map to municipality names
municipality_max = max_idx
municipality_min = min_idx

# Get the actual values
max_value = foreign_values[max_idx]
min_value = foreign_values[min_idx]

# Compute difference
difference = max_value - min_value

print(f"Final Answer: {municipality_max}, {municipality_min}, {difference:.1f}")