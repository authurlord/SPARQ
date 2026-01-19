import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, coercing errors to NaN if necessary
df['area in sqkm'] = pd.to_numeric(df['area in sqkm'], errors='coerce')
df['population density 2012 (km 2)'] = pd.to_numeric(df['population density 2012 (km 2)'], errors='coerce')

# Drop rows with NaN values in either column
df.dropna(subset=['area in sqkm', 'population density 2012 (km 2)'], inplace=True)

# Calculate correlation coefficient
correlation = df['area in sqkm'].corr(df['population density 2012 (km 2)'])

print(f"Final Answer: {correlation:.4f}")