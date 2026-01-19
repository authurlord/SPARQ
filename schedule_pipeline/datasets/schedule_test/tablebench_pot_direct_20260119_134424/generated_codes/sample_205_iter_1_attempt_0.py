import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, forcing errors to be coerced
df['Temperature T (°C)'] = pd.to_numeric(df['Temperature T (°C)'], errors='coerce')
df['Density of air ρ (kg/m3)'] = pd.to_numeric(df['Density of air ρ (kg/m3)'], errors='coerce')
df['Speed of sound c (m/s)'] = pd.to_numeric(df['Speed of sound c (m/s)'], errors='coerce')

# Calculate correlation coefficients
corr_temp = df['Speed of sound c (m/s)'].corr(df['Temperature T (°C)'])
corr_density = df['Speed of sound c (m/s)'].corr(df['Density of air ρ (kg/m3)'])

# Determine which has greater influence based on absolute correlation
if abs(corr_temp) > abs(corr_density):
    influence = "Temperature T (°C)"
else:
    influence = "Density of air ρ (kg/m3)"

print(f"Final Answer: {influence}")