import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'total' row since it's a summary
df_filtered = df[df['ecozone'] != 'total']

# Extract the two columns for correlation
territorial_area = df_filtered['area (km square) territorial waters']
total_area_percentage = df_filtered['percentage of total area (foreez)']

# Calculate the correlation coefficient
correlation = territorial_area.corr(total_area_percentage)
print(f"Final Answer: {correlation:.3f}")