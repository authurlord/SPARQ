import pandas as pd

df = pd.read_csv('table.csv')
# Select the relevant columns for analysis
energy = df['Energy (kJ)']
carbs = df['Carbohydrates (g)']
fat = df['Fat (g)']
protein = df['Protein (g)']

# Calculate correlation coefficients
corr_carbs = energy.corr(carbs)
corr_fat = energy.corr(fat)
corr_protein = energy.corr(protein)

# Determine which has the highest correlation
if corr_carbs >= corr_fat and corr_carbs >= corr_protein:
    result = "carbohydrates"
elif corr_fat >= corr_carbs and corr_fat >= corr_protein:
    result = "fat"
else:
    result = "protein"

print(f"Final Answer: {result}")