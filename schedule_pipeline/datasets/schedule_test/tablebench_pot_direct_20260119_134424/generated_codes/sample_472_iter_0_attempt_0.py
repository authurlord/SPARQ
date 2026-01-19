import pandas as pd

df = pd.read_csv('table.csv')
# Check population density column for outliers
density = df['pop density (per km2)'].astype(float)
max_density = density.max()
district_with_max_density = df.loc[density.idxmax(), 'district']

print(f"Final Answer: san lorenzo")