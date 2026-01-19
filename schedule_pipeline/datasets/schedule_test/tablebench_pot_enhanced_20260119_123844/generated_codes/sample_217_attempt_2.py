import pandas as pd

df = pd.read_csv('table.csv')

# Extract the relevant columns for analysis
energy = df['Energy (kJ)'].astype(float)
carbs = df['Carbohydrates (g)'].astype(float)
fat = df['Fat (g)'].astype(float)
protein = df['Protein (g)'].astype(float)

# Calculate correlations
corr_carbs = energy.corr(carbs)
corr_fat = energy.corr(fat)
corr_protein = energy.corr(protein)

# Determine which has the highest correlation
max_corr = max(corr_carbs, corr_fat, corr_protein)
if max_corr == corr_carbs:
    result = "Carbohydrates"
elif max_corr == corr_fat:
    result = "Fat"
else:
    result = "Protein"

print(f"Final Answer: {result}")