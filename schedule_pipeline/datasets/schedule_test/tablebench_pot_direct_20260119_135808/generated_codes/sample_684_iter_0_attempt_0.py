import pandas as pd

df = pd.read_csv('table.csv')
# Clean the data: remove the row with invalid year '191822'
df_clean = df[df['year'] != '191822']
# Convert 'typhus' and 'smallpox' to numeric, coercing errors
df_clean['typhus'] = pd.to_numeric(df_clean['typhus'], errors='coerce')
df_clean['smallpox'] = pd.to_numeric(df_clean['smallpox'], errors='coerce')
# Drop rows with NaN values after conversion
df_clean.dropna(subset=['typhus', 'smallpox'], inplace=True)
# Calculate correlation coefficient
correlation = df_clean['typhus'].corr(df_clean['smallpox'])
print(f"Final Answer: {correlation:.3f}")