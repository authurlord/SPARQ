import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert the 'foreign' column to numeric (it's stored as strings with decimal points)
# We need to extract the row where language is 'foreign'
foreign_data = df[df['language'] == 'foreign']

# Convert the values in the foreign row to numeric
foreign_values = pd.to_numeric(foreign_data.iloc[:, 1:], errors='coerce')

# Find the municipality with the highest and lowest values
max_value = foreign_values.max()
min_value = foreign_values.min()
difference = max_value - min_value

# Get the corresponding municipality names
municipalities = foreign_values.index.tolist()
max_mun = municipalities[foreign_values.idxmax()]
min_mun = municipalities[foreign_values.idxmin()]

print(f"Final Answer: {max_mun}, {min_mun}, {difference:.1f}")