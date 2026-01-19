import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'population' and 'density' columns to numeric
df['population'] = pd.to_numeric(df['population'])
df['density'] = pd.to_numeric(df['density'])

# Calculate correlation coefficient between 'population' and 'density'
correlation = df['population'].corr(df['density'])
print(f"Final Answer: {correlation:.4f}")