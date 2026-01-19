import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation between 'area ( has )' and 'pop density (per km2)'
correlation = df['area ( has )'].corr(df['pop density (per km2)'])
print(f"Final Answer: {correlation:.3f}")