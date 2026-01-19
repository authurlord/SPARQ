import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric for calculation
speed_of_sound = pd.to_numeric(df['Speed of sound c (m/s)'])
density_of_air = pd.to_numeric(df['Density of air ρ (kg/m3)'])

# Calculate correlation coefficient
correlation = speed_of_sound.corr(density_of_air)
print(f"Final Answer: {correlation:.4f}")