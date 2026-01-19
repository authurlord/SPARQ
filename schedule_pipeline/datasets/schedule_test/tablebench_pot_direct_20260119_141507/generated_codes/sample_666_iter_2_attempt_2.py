import pandas as pd

df = pd.read_csv('table.csv')
# Correct column names as provided in the table
area_column = 'area ( has )'
density_column = 'pop density (per km2)'

# Calculate the correlation coefficient
correlation = df[area_column].corr(df[density_column])
print(f"Final Answer: {correlation:.3f}")