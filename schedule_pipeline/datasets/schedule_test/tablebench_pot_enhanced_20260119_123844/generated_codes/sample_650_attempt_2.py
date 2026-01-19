import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric for calculation
area = pd.to_numeric(df['area in sqkm'])
density = pd.to_numeric(df['population density 2012 (km 2 )'])

# Calculate correlation coefficient
correlation = area.corr(density)
print(f"Final Answer: {correlation:.4f}")