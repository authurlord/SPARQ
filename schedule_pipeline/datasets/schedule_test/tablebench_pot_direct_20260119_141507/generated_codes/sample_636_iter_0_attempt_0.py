import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'area (km square)' and 'gdp (billion us)'
correlation_coefficient = df['area (km square)'].corr(df['gdp (billion us)'])
print(f"Final Answer: {correlation_coefficient:.3f}")