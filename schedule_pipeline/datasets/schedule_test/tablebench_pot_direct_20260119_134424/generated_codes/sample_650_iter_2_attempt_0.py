import pandas as pd

df = pd.read_csv('table.csv')
# Correct column name handling with spaces and parentheses
area = df['area in sqkm'].astype(float)
density = df['population density 2012 (km 2)'].astype(float)
# Calculate correlation coefficient
correlation = area.corr(density)
print(f"Final Answer: {correlation:.4f}")