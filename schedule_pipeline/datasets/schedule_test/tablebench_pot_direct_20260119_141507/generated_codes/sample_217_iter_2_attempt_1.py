import pandas as pd
import numpy as np

# Load the table
df = pd.read_csv('table.csv')

# Select only the relevant columns
energy_col = 'Energy (kJ)'
carbs_col = 'Carbohydrates (g)'
fat_col = 'Fat (g)'
protein_col = 'Protein (g)'

# Extract the numerical values for these columns
energy = df[energy_col].dropna().astype(float)
carbs = df[carbs_col].dropna().astype(float)
fat = df[fat_col].dropna().astype(float)
protein = df[protein_col].dropna().astype(float)

# Calculate correlation coefficients
corr_carbs = energy.corr(carbs)
corr_fat = energy.corr(fat)
corr_protein = energy.corr(protein)

# Find the highest correlation
max_corr = max(corr_carbs, corr_fat, corr_protein)
if max_corr == corr_carbs:
    result = "carbohydrates"
elif max_corr == corr_fat:
    result = "fat"
else:
    result = "protein"

print(f"Final Answer: {result}")