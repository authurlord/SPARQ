import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert '% change' to numeric, removing the '-' sign and converting to float
df['% change'] = df['% change'].str.replace(' - ', '-').astype(float)
# Calculate correlation between 'density (pop / km square)' and '% change'
correlation = df['density (pop / km square)'].corr(df['% change'])
print(f"Final Answer: {correlation:.4f}")