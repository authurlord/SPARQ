import pandas as pd

df = pd.read_csv('table.csv')
# Convert elevation and prominence to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'])
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'])

# Calculate correlation coefficient
correlation = df['elevation (m)'].corr(df['prominence (m)'])

print(f"Final Answer: {correlation:.3f}")