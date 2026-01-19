import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['area in sqkm'] = pd.to_numeric(df['area in sqkm'])
df['population density 2012 (km 2)'] = pd.to_numeric(df['population density 2012 (km 2)'])

# Calculate correlation coefficient
correlation = df['area in sqkm'].corr(df['population density 2012 (km 2)'])
print(f"Final Answer: {correlation:.4f}")