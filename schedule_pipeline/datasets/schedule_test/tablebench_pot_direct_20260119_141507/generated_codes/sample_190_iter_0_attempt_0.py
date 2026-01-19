import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric (handle string values like '39.13')
df['runs scored'] = pd.to_numeric(df['runs scored'], errors='coerce')
df['balls faced'] = pd.to_numeric(df['balls faced'], errors='coerce')
df['average'] = pd.to_numeric(df['average'].str.replace('%', ''), errors='coerce')
df['sr'] = pd.to_numeric(df['sr'].str.replace('%', ''), errors='coerce')

# Compute correlation with 'average' and 'sr'
correlations = df[['innings', 'runs scored', 'balls faced']].corrwith(df[['average', 'sr']])

# Get the top 2 factors by absolute correlation (combined influence)
correlation_abs = correlations.abs()
top_factors = correlation_abs.sort_values(ascending=False).head(2).index.tolist()

print(f"Final Answer: {top_factors[0]}, {top_factors[1]}")