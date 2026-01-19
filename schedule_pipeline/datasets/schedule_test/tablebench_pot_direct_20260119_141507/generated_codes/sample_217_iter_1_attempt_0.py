import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row which is headers like "STAPLE:", "Component (per 100g portion)"
# The first row is actually the header row for the components, so we skip it
# The actual data starts from the second row (index 1)

# Extract the data rows (skip the first row)
data_rows = df.iloc[1:]

# Select only the relevant columns
energy_col = 'Energy (kJ)'
carbs_col = 'Carbohydrates (g)'
fat_col = 'Fat (g)'
protein_col = 'Protein (g)'

# Convert to numeric (some values may be missing, e.g., '-', so we handle that)
data_rows['Energy (kJ)'] = pd.to_numeric(data_rows['Energy (kJ)'], errors='coerce')
data_rows['Carbohydrates (g)'] = pd.to_numeric(data_rows['Carbohydrates (g)'], errors='coerce')
data_rows['Fat (g)'] = pd.to_numeric(data_rows['Fat (g)'], errors='coerce')
data_rows['Protein (g)'] = pd.to_numeric(data_rows['Protein (g)'], errors='coerce')

# Drop rows with NaN values
data_rows = data_rows.dropna()

# Compute correlation between energy and each nutrient
correlation_carbs = data_rows[energy_col].corr(data_rows[carbs_col])
correlation_fat = data_rows[energy_col].corr(data_rows[fat_col])
correlation_protein = data_rows[energy_col].corr(data_rows[protein_col])

# Find the nutrient with the highest absolute correlation
max_corr = max(abs(correlation_carbs), abs(correlation_fat), abs(correlation_protein))
if max_corr == abs(correlation_carbs):
    answer = "carbohydrates"
elif max_corr == abs(correlation_fat):
    answer = "fat"
else:
    answer = "protein"

print(f"Final Answer: {answer}")