import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Select relevant columns: Energy (kJ), Carbohydrates (g), Fat (g), Protein (g)
energy_col = 'Energy (kJ)'
carbs_col = 'Carbohydrates (g)'
fat_col = 'Fat (g)'
protein_col = 'Protein (g)'

# Drop the first row (header row) and convert to numeric (handle missing values like '-')
df_clean = df.drop(0).copy()
df_clean = df_clean[[energy_col, carbs_col, fat_col, protein_col]].apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN values due to missing data (e.g., '-')
df_clean = df_clean.dropna()

# Compute correlation between energy and each macronutrient
correlation_carbs = df_clean[energy_col].corr(df_clean[carbs_col])
correlation_fat = df_clean[energy_col].corr(df_clean[fat_col])
correlation_protein = df_clean[energy_col].corr(df_clean[protein_col])

# Find the highest correlation
max_corr = max(correlation_carbs, correlation_fat, correlation_protein)
if max_corr == correlation_carbs:
    answer = "carbohydrates"
elif max_corr == correlation_fat:
    answer = "fat"
else:
    answer = "protein"

print(f"Final Answer: {answer}")