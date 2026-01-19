import pandas as pd

df = pd.read_csv('table.csv')

# Filter the row where language is 'foreign'
foreign_row = df[df['language'] == 'foreign']

# Extract the values for each municipality
foreign_values = foreign_row.iloc[0][1:].astype(float)

# Find the municipality with the highest and lowest values
max_value = foreign_values.max()
min_value = foreign_values.min()

# Get the corresponding municipality names
municipalities = df.columns[1:]  # All municipality columns
max_mun = municipalities[foreign_values.idxmax()]
min_mun = municipalities[foreign_values.idxmin()]

difference = max_value - min_value

print(f"Final Answer: {max_mun}, {difference:.1f}")