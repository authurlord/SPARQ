import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['Speed of sound c (m/s)'] = pd.to_numeric(df['Speed of sound c (m/s)'])
df['Density of air ρ (kg/m3)'] = pd.to_numeric(df['Density of air ρ (kg/m3)'])

# Calculate correlation coefficient
correlation = df['Speed of sound c (m/s)'].corr(df['Density of air ρ (kg/m3)'])
print(f"Final Answer: {correlation:.4f}")