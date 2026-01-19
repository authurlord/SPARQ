import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert '% change' column to numeric, removing the '-' sign and converting to float
df['% change'] = df['% change'].str.replace(' - ', '-').astype(float)
# Convert 'density (pop / km square)' to numeric
df['density (pop / km square)'] = pd.to_numeric(df['density (pop / km square)'])
# Calculate correlation coefficient
correlation = df['density (pop / km square)'].corr(df['% change'])
print(f"Final Answer: {correlation:.4f}")