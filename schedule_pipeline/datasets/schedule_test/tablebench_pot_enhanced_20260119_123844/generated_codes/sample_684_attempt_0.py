import pandas as pd

df = pd.read_csv('table.csv')
# Drop the row with non-numeric smallpox data
df = df[df['smallpox'] != '2940 (avg)']
# Convert 'typhus' and 'smallpox' to numeric
df['typhus'] = pd.to_numeric(df['typhus'])
df['smallpox'] = pd.to_numeric(df['smallpox'])
# Calculate correlation coefficient
correlation = df['typhus'].corr(df['smallpox'])
print(f"Final Answer: {correlation:.4f}")