import pandas as pd

df = pd.read_csv('table.csv')
# Correct column name is 'area (ha)', not 'area ( has)'
correlation = df['area (ha)'].corr(df['pop density (per km2)'])
print(f"Final Answer: {correlation:.3f}")