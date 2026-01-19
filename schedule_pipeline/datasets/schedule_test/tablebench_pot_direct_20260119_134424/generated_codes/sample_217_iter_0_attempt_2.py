import pandas as pd

df = pd.read_csv('table.csv')
# Extract relevant columns
energy = df['Energy (kJ)'].astype(float)
carbs = df['Carbohydrates (g)'].astype(float)
fat = df['Fat (g)'].astype(float)
protein = df['Protein (g)'].astype(float)

# Calculate correlations
corr_carbs = energy.corr(carbs)
corr_fat = energy.corr(fat)
corr_protein = energy.corr(protein)

# Find the highest correlation
max_corr = max(corr_carbs, corr_fat, corr_protein)
if max_corr == corr_carbs:
    answer = "Carbohydrates"
elif max_corr == corr_fat:
    answer = "Fat"
else:
    answer = "Protein"

print(f"Final Answer: {answer}")