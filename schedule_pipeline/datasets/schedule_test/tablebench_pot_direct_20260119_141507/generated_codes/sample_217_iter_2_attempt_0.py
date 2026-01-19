import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select only the relevant columns: 'Energy (kJ)', 'Carbohydrates (g)', 'Fat (g)', 'Protein (g)'
# Drop the first row which is the header "Component (per 100g portion)"
# The actual data starts from the second row

# We'll extract the data rows after the first row (index 1 onwards)
data_rows = df.iloc[1:].copy()

# Select only the needed columns
energy = data_rows['Energy (kJ)'].astype(float)
carbs = data_rows['Carbohydrates (g)'].astype(float)
fat = data_rows['Fat (g)'].astype(float)
protein = data_rows['Protein (g)'].astype(float)

# Calculate correlation between energy and each nutrient
corr_carbs = energy.corr(carbs)
corr_fat = energy.corr(fat)
corr_protein = energy.corr(protein)

# Find the nutrient with the highest absolute correlation
max_corr = max(abs(corr_carbs), abs(corr_fat), abs(corr_protein))
if max_corr == abs(corr_carbs):
    result = "carbohydrates"
elif max_corr == abs(corr_fat):
    result = "fat"
else:
    result = "protein"

print(f"Final Answer: {result}")