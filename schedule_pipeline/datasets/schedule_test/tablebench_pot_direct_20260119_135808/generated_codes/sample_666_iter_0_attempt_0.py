import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['area ( has )'] = pd.to_numeric(df['area ( has )'])
df['pop density (per km2)'] = pd.to_numeric(df['pop density (per km2)'])

# Calculate correlation coefficient
correlation = df['area ( has )'].corr(df['pop density (per km2)'])

print(f"Final Answer: {correlation:.4f}")