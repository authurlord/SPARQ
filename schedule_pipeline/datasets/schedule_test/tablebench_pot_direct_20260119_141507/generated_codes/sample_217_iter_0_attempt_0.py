import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select only the relevant columns for analysis
# Drop the first row (STAPLE:) as it's a header
data_rows = df.iloc[1:, :]  # Skip the first row which is headers

# Extract the columns for energy and macronutrients
energy = data_rows['Energy (kJ)']
carbs = data_rows['Carbohydrates (g)']
fat = data_rows['Fat (g)']
protein = data_rows['Protein (g)']

# Calculate correlation with energy
corr_carbs = energy.corr(carbs)
corr_fat = energy.corr(fat)
corr_protein = energy.corr(protein)

# Find the nutrient with the highest absolute correlation
max_corr = max(abs(corr_carbs), abs(corr_fat), abs(corr_protein))
if max_corr == abs(corr_carbs):
    answer = "carbohydrates"
elif max_corr == abs(corr_fat):
    answer = "fat"
else:
    answer = "protein"

print(f"Final Answer: {answer}")