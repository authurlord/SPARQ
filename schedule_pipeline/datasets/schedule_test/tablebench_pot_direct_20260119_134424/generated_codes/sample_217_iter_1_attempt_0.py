import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns
energy = df['Energy (kJ)'].astype(float)
carbs = df['Carbohydrates (g)'].astype(float)
fat = df['Fat (g)'].astype(float)
protein = df['Protein (g)'].astype(float)

# Calculate correlations
correlation_carbs = energy.corr(carbs)
correlation_fat = energy.corr(fat)
correlation_protein = energy.corr(protein)

# Find the nutrient with the highest correlation
max_corr = max(correlation_carbs, correlation_fat, correlation_protein)
if max_corr == correlation_carbs:
    result = "Carbohydrates"
elif max_corr == correlation_fat:
    result = "Fat"
else:
    result = "Protein"

print(f"Final Answer: {result}")