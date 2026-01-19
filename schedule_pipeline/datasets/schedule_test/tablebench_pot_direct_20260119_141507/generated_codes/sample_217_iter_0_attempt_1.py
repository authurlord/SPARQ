import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select relevant columns: 'Energy (kJ)', 'Carbohydrates (g)', 'Fat (g)', 'Protein (g)'
energy = df['Energy (kJ)'].dropna()
carbs = df[['Carbohydrates (g)']].dropna()['Carbohydrates (g)']
fat = df[['Fat (g)']].dropna()['Fat (g)']
protein = df[['Protein (g)']].dropna()['Protein (g)']

# Ensure all values are numeric and handle missing values (e.g., '-')
carbs = pd.to_numeric(carbs, errors='coerce')
fat = pd.to_numeric(fat, errors='coerce')
protein = pd.to_numeric(protein, errors='coerce')
energy = pd.to_numeric(energy, errors='coerce')

# Drop rows with NaN after conversion
energy = energy.dropna()
carbs = carbs.dropna()
fat = fat.dropna()
protein = protein.dropna()

# Compute correlation coefficients
corr_carbs = energy.corr(carbs)
corr_fat = energy.corr(fat)
corr_protein = energy.corr(protein)

# Find the highest correlation
max_corr = max(corr_carbs, corr_fat, corr_protein)
if max_corr == corr_carbs:
    final_answer = "carbohydrates"
elif max_corr == corr_fat:
    final_answer = "fat"
else:
    final_answer = "protein"

print(f"Final Answer: {final_answer}")