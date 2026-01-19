import pandas as pd

df = pd.read_csv('table.csv')
# Convert the columns to numeric type for calculation
area_territorial = pd.to_numeric(df['area (km square) territorial waters'], errors='coerce')
percentage_total = pd.to_numeric(df['percentage of total area (foreez)'], errors='coerce')

# Calculate the correlation coefficient
correlation = area_territorial.corr(percentage_total)
print(f"Final Answer: {correlation:.4f}")