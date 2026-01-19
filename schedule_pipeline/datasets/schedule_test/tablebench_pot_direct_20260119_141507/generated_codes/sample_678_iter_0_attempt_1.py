import pandas as pd

df = pd.read_csv('table.csv')

# Convert '% change' to numeric by removing the '%' and converting to float
df['% change'] = df['% change'].str.replace('%', '').astype(float)

# Calculate the correlation between 'density (pop / km square)' and '% change'
correlation = df['density (pop / km square)'].corr(df['% change'])

print(f"Final Answer: {correlation:.3f}")