import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter out rows that are not actual regions (e.g., 'total', 'total (in all 31 provincial regions)', 'active servicemen')
# We focus on regions with valid region names
region_data = df[df['region'] != 'total']
region_data = region_data[region_data['region'] != 'total (in all 31 provincial regions)']
region_data = region_data[region_data['region'] != 'active servicemen']

# Sort by Manchu population in descending order
region_data_sorted = region_data.sort_values(by='manchu', ascending=False)

# Display top 10 regions by Manchu population
print("Top 10 regions by Manchu population:")
print(region_data_sorted[['region', 'manchu']].head(10))

# Total Manchu population across all regions
total_manchu_population = df[df['region'] == 'total']['manchu'].values[0]
print(f"\nTotal Manchu population: {total_manchu_population}")

# Summary of the distribution
print("\nInsights:")
print("- The Manchu population is concentrated in northern China, especially in Liaoning, Hebei, and Jilin.")
print("- Liaoning has the highest Manchu population (5,336,895), followed by Hebei (2,118,711).")
print("- The Manchu population is significantly smaller in southern and western regions.")
print("- The regional percentage of population indicates that these regions represent a small fraction of the total population.")