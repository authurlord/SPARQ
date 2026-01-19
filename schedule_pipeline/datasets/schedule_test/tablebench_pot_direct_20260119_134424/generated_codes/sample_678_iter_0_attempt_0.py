import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% change' to numeric, handling the '-' sign
df['% change'] = pd.to_numeric(df['% change'].str.replace(' - ', '-'), errors='coerce')
# Calculate correlation between 'density (pop / km square)' and '% change'
correlation = df['density (pop / km square)'].corr(df['% change'])
print(f"Final Answer: {correlation:.4f}")