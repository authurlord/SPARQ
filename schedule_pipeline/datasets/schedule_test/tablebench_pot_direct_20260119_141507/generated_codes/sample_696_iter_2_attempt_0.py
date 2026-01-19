import pandas as pd

df = pd.read_csv('table.csv')

# Convert the 'foreign' column to numeric, handling any non-numeric entries
foreign_values = pd.to_numeric(df['foreign'], errors='coerce')

# Remove any NaN values (if any)
foreign_values = foreign_values.dropna()

# Find the maximum and minimum values
max_foreign = foreign_values.max()
min_foreign = foreign_values.min()

# Find the corresponding municipality names
# The municipalities are the column names
municipalities = df.columns[1:]  # Skip 'language'

# Get the index of max and min values
max_idx = foreign_values.idxmax()
min_idx = foreign_values.idxmin()

# Map to municipality name
max_muni = municipalities[max_idx]
min_muni = municipalities[min_idx]

# Calculate the difference
difference = max_foreign - min_foreign

print(f"Final Answer: {max_muni}, {min_muni}, {difference:.1f}")