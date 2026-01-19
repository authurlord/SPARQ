import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' and 'smallpox' to numeric, handling non-numeric values like 'avg'
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['smallpox'] = pd.to_numeric(df['smallpox'].str.replace(r'\(avg\)', '', regex=True), errors='coerce')

# Drop rows with missing values after conversion
df.dropna(subset=['typhus', 'smallpox'], inplace=True)

# Calculate correlation coefficient
correlation = df['typhus'].corr(df['smallpox'])

print(f"Final Answer: {correlation:.4f}")