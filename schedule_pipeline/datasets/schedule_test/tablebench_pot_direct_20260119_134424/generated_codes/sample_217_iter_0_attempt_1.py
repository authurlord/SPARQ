import pandas as pd

df = pd.read_csv('table.csv')

# Drop the first row which contains headers for the data
df = df.drop(0)

# Convert relevant columns to numeric
df['Energy (kJ)'] = pd.to_numeric(df['Energy (kJ)'], errors='coerce')
df['Carbohydrates (g)'] = pd.to_numeric(df['Carbohydrates (g)'], errors='coerce')
df['Fat (g)'] = pd.to_numeric(df['Fat (g)'], errors='coerce')
df['Protein (g)'] = pd.to_numeric(df['Protein (g)'], errors='coerce')

# Calculate correlation with energy
correlation_carbs = df['Energy (kJ)'].corr(df['Carbohydrates (g)'])
correlation_fat = df['Energy (kJ)'].corr(df['Fat (g)'])
correlation_protein = df['Energy (kJ)'].corr(df['Protein (g)'])

# Find which has the highest correlation
max_corr = max(correlation_carbs, correlation_fat, correlation_protein)
if max_corr == correlation_carbs:
    result = "Carbohydrates"
elif max_corr == correlation_fat:
    result = "Fat"
else:
    result = "Protein"

print(f"Final Answer: {result}")