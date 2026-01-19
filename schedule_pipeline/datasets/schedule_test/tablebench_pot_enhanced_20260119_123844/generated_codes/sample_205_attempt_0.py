import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['Temperature T (°C)'] = pd.to_numeric(df['Temperature T (°C)'])
df['Speed of sound c (m/s)'] = pd.to_numeric(df['Speed of sound c (m/s)'])
df['Density of air ρ (kg/m3)'] = pd.to_numeric(df['Density of air ρ (kg/m3)'])

# Calculate correlation coefficients
corr_temp_c = df['Temperature T (°C)'].corr(df['Speed of sound c (m/s)'])
corr_density_c = df['Density of air ρ (kg/m3)'].corr(df['Speed of sound c (m/s)'])

# Compare absolute values of correlations
if abs(corr_temp_c) > abs(corr_density_c):
    influence = "Temperature T (°C)"
else:
    influence = "Density of air ρ (kg/m3)"

print(f"Final Answer: {influence}")