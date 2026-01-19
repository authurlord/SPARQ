import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' and 'smallpox' columns to numeric, handling any non-numeric values
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['smallpox'] = pd.to_numeric(df['smallpox'], errors='coerce')

# Calculate the correlation coefficient
correlation = df['typhus'].corr(df['smallpox'])

print(f"Final Answer: {correlation:.3f}")