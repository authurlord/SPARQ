import pandas as pd

df = pd.read_csv('table.csv')
# Convert '% change' column to numeric, handling any potential errors
df['% change'] = pd.to_numeric(df['% change'].str.replace('%', ''), errors='coerce')

# Drop rows with NaN due to invalid % change values
df = df.dropna(subset=['% change', 'density (pop / km square)'])

# Calculate the correlation between 'density (pop / km square)' and '% change'
correlation = df['density (pop / km square)'].corr(df['% change'])
print(f"Final Answer: {correlation:.3f}")