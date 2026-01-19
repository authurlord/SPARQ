import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation between 'Speed of sound c (m/s)' and 'Density of air ρ (kg/m3)'
correlation = df['Speed of sound c (m/s)'].astype(float).corr(df['Density of air ρ (kg/m3)'].astype(float))
print(f"Final Answer: {correlation:.4f}")