import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Select only the relevant columns
columns_of_interest = ['Energy (kJ)', 'Carbohydrates (g)', 'Fat (g)', 'Protein (g)']

# Filter out rows where any of the nutrient values is missing (e.g., '-')
df_filtered = df[columns_of_interest].dropna()

# Compute correlation between energy and each nutrient
correlations = df_filtered.corr()['Energy (kJ)'].abs()

# Find the nutrient with the highest correlation
most_related_nutrient = correlations.idxmax()

print(f"Final Answer: {most_related_nutrient}")