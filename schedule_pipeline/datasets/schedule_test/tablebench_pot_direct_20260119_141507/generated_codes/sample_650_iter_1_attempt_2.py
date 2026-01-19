import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation between 'area in sqkm' and 'population density 2012 (km 2)'
correlation = df['area in sqkm'].corr(df['population density 2012 (km 2)'])
print(f"Final Answer: {correlation:.3f}")