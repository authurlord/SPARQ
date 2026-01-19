import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total freshwater withdrawal' and 'per capita withdrawal' to numeric
df['total freshwater withdrawal'] = pd.to_numeric(df['total freshwater withdrawal'])
df['per capita withdrawal'] = pd.to_numeric(df['per capita withdrawal'])

# Calculate correlation coefficient
correlation = df['total freshwater withdrawal'].corr(df['per capita withdrawal'])
print(f"Final Answer: {correlation:.4f}")