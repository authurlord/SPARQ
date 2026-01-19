import pandas as pd

df = pd.read_csv('table.csv')

# Drop the first row which contains headers
df = df.drop(0)

# Convert all relevant columns to numeric
energy = pd.to_numeric(df['Energy (kJ)'], errors='coerce')
carbs = pd.to_numeric(df['Carbohydrates (g)'], errors='coerce')
fat = pd.to_numeric(df['Fat (g)'], errors='coerce')
protein = pd.to_numeric(df['Protein (g)'], errors='coerce')

# Calculate correlations
corr_carbs = energy.corr(carbs)
corr_fat = energy.corr(fat)
corr_protein = energy.corr(protein)

# Find the highest correlation
max_corr = max(corr_carbs, corr_fat, corr_protein)
if max_corr == corr_carbs:
    result = "Carbohydrates"
elif max_corr == corr_fat:
    result = "Fat"
else:
    result = "Protein"

print(f"Final Answer: {result}")