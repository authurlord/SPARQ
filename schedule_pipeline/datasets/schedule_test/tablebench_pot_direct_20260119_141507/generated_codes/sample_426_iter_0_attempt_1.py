import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where 'percentage in manchu population' is above a threshold (e.g., > 1%) to focus on significant concentrations
high_percentage_regions = df[df['percentage in manchu population'] > 1.0]

# Sort by Manchu population in descending order to highlight the main contributors
high_percentage_regions_sorted = high_percentage_regions.sort_values(by='manchu', ascending=False)

print("Main components of the table:")
print("Columns: region, total population, manchu, percentage in manchu population, regional percentage of population")
print("\nInsights on Manchu population distribution:")
print(high_percentage_regions_sorted[['region', 'manchu', 'percentage in manchu population']].to_string(index=False))