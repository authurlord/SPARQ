import pandas as pd

df = pd.read_csv('table.csv')
# Extract the two columns for correlation
speed_of_sound = df['Speed of sound c (m/s)']
density_of_air = df['Density of air ρ (kg/m3)']

# Calculate the correlation coefficient
correlation = speed_of_sound.astype(float).corr(density_of_air.astype(float))

print(f"Final Answer: {correlation:.4f}")