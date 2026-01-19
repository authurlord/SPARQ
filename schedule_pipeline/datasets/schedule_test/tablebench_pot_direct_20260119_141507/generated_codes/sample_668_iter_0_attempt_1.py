import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'total' row as it is not an ecozone
df_filtered = df[df['ecozone'] != 'total']

# Extract the relevant columns
territorial_area = df_filtered['area (km square) territorial waters']
percent_total_area = df_filtered['percentage of total area (foreez)']

# Calculate the correlation coefficient
correlation_coefficient = territorial_area.corr(percent_total_area)

print(f"Final Answer: {correlation_coefficient:.3f}")