import pandas as pd

df = pd.read_csv('table.csv')

# Convert Temperature T (°C) to Kelvin for better analysis (though not strictly needed for correlation)
# Correlation between speed of sound and temperature
corr_temp_c = df['Speed of sound c (m/s)'].corr(df['Temperature T (°C)'])

# Correlation between speed of sound and density
corr_density = df['Speed of sound c (m/s)'].corr(df['Density of air ρ (kg/m3)'])

# Determine which has greater absolute correlation
if abs(corr_temp_c) > abs(corr_density):
    result = "Temperature T (°C)"
else:
    result = "Density of air ρ (kg/m3)"

print(f"Final Answer: {result}")