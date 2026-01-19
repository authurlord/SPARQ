import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Magnitude' and 'Depth (km)' to numeric
df['Magnitude'] = pd.to_numeric(df['Magnitude'])
df['Depth (km)'] = pd.to_numeric(df['Depth (km)'])

# Calculate correlation coefficient
correlation = df['Magnitude'].corr(df['Depth (km)'])
print(f"Final Answer: {correlation:.3f}")