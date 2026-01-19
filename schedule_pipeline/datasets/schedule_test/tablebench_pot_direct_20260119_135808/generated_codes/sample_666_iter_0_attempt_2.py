import pandas as pd

df = pd.read_csv('table.csv')
# Convert area from hectares to square kilometers
df['area (km2)'] = df['area ( has )'] * 0.01
# Calculate correlation between area (km2) and pop density (per km2)
correlation = df['area (km2)'].corr(df['pop density (per km2)'])
print(f"Final Answer: {correlation:.4f}")