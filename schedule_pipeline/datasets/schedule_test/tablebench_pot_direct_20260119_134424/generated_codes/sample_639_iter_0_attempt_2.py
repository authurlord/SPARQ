import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'area (km 2 )' and 'population' to numeric, handling any potential non-numeric values
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Drop rows with missing values in either column
df.dropna(subset=['area (km 2 )', 'population'], inplace=True)

# Calculate the correlation coefficient
correlation = df['area (km 2 )'].corr(df['population'])

print(f"Final Answer: {correlation:.4f}")