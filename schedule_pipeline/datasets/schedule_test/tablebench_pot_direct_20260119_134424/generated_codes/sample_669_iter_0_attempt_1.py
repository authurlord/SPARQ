import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['area (km square)'] = pd.to_numeric(df['area (km square)'])
df['pop'] = pd.to_numeric(df['pop'])

# Calculate correlation coefficient
correlation = df['area (km square)'].corr(df['pop'])
print(f"Final Answer: {correlation:.4f}")