import pandas as pd

df = pd.read_csv('table.csv')
# Convert the relevant columns to numeric for calculation
area_territorial = pd.to_numeric(df['area (km square) territorial waters'], errors='coerce')
percentage_total = pd.to_numeric(df['percentage of total area (foreez)'], errors='coerce')

# Drop rows with missing values
data_clean = pd.DataFrame({'area': area_territorial, 'percentage': percentage_total}).dropna()

# Calculate correlation coefficient
correlation = data_clean['area'].corr(data_clean['percentage'])

print(f"Final Answer: {correlation:.4f}")