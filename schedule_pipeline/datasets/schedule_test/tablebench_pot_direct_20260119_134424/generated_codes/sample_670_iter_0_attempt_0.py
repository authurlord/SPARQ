import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['total freshwater withdrawal'] = pd.to_numeric(df['total freshwater withdrawal'], errors='coerce')
df['per capita withdrawal'] = pd.to_numeric(df['per capita withdrawal'], errors='coerce')

# Calculate correlation coefficient
correlation = df['total freshwater withdrawal'].corr(df['per capita withdrawal'])
print(f"Final Answer: {correlation:.4f}")