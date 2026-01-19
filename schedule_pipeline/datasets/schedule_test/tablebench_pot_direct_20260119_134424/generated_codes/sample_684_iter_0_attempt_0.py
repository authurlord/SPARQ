import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric, handling non-numeric values like 'avg'
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['smallpox'] = pd.to_numeric(df['smallpox'], errors='coerce')

# Drop rows with NaN values in either column
df.dropna(subset=['typhus', 'smallpox'], inplace=True)

# Calculate correlation coefficient
correlation = df['typhus'].corr(df['smallpox'])
print(f"Final Answer: {correlation:.3f}")