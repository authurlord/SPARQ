import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between 'Speed of sound c (m/s)' and 'Density of air ρ (kg/m3)'
correlation = df['Speed of sound c (m/s)'].corr(df['Density of air ρ (kg/m3)'])
print(f"Final Answer: {correlation:.3f}")