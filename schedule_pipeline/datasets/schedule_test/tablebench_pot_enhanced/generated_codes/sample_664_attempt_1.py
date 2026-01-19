import pandas as pd

df = pd.read_csv('table.csv')
# Convert the relevant columns to float for numerical operations
speed_of_sound = df['Speed of sound c (m/s)'].astype(float)
density_of_air = df['Density of air ρ (kg/m3)'].astype(float)

# Calculate the correlation coefficient
correlation = speed_of_sound.corr(density_of_air)
print(f"Final Answer: {correlation:.4f}")