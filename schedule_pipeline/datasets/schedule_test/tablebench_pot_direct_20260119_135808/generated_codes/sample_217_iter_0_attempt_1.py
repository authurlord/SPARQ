import pandas as pd

df = pd.read_csv('table.csv')
# Select the relevant columns for analysis
energy = df['Energy (kJ)']
carbs = df['Carbohydrates (g)']
fat = df['Fat (g)']
protein = df['Protein (g)']

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