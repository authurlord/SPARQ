import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'area (ha)' and 'pop density (per km²)'
correlation = df['area (ha)'].corr(df['pop density (per km2)'])
print(f"Final Answer: {correlation:.3f}")