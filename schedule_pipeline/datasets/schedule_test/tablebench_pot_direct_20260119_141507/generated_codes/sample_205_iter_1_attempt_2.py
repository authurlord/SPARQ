import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert temperature from °C to Kelvin for proper physical interpretation
df['Temperature T (°C)'] = df['Temperature T (°C)'].astype(float)
df['Speed of sound c (m/s)'] = df['Speed of sound c (m/s)'].astype(float)
df['Density of air ρ (kg/m3)'] = df['Density of air ρ (kg/m3)'].astype(float)

# Compute correlation between speed of sound and temperature
corr_temp_c = df['Speed of sound c (m/s)'].corr(df['Temperature T (°C)'])

# Compute correlation between speed of sound and density
corr_density_c = df['Speed of sound c (m/s)'].corr(df['Density of air ρ (kg/m3)'])

# Compare absolute values to determine which has greater influence
if abs(corr_temp_c) > abs(corr_density_c):
    result = 'temperature T (°C)'
else:
    result = 'density of air ρ (kg/m³)'

print(f"Final Answer: {result}")