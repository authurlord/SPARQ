import pandas as pd

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['Temperature T (°C)'] = pd.to_numeric(df['Temperature T (°C)'])
df['Speed of sound c (m/s)'] = pd.to_numeric(df['Speed of sound c (m/s)'])
df['Density of air ρ (kg/m3)'] = pd.to_numeric(df['Density of air ρ (kg/m3)'])

# Calculate correlation coefficients
corr_temp = df['Speed of sound c (m/s)'].corr(df['Temperature T (°C)'])
corr_density = df['Speed of sound c (m/s)'].corr(df['Density of air ρ (kg/m3)'])

# Compare correlations
if abs(corr_temp) > abs(corr_density):
    influence = 'Temperature T (°C)'
else:
    influence = 'Density of air ρ (kg/m3)'

print(f"Final Answer: {influence}")