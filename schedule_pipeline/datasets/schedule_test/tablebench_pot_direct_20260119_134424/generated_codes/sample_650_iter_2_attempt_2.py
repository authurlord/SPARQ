import pandas as pd

df = pd.read_csv('table.csv')
# Correct column names with proper handling of spaces and parentheses
area = df['area in sqkm']
density = df['population density 2012 (km 2)']

# Calculate correlation coefficient
correlation = area.corr(density)
print(f"Final Answer: {correlation:.4f}")