import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant columns
columns_of_interest = ['Energy (kJ)', 'Carbohydrates (g)', 'Fat (g)', 'Protein (g)']
df_filtered = df[columns_of_interest]

# Drop any rows with NaN values (e.g., due to missing data like '-')
df_filtered = df_filtered.dropna()

# Compute correlation between Energy and each nutrient
correlations = df_filtered.corr()['Energy (kJ)'].abs()

# Find the nutrient with the highest correlation
most_correlated_nutrient = correlations.idxmax()

print(f"Final Answer: {most_correlated_nutrient}")