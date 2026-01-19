import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' and 'typhoid fever' columns to numeric, coercing errors to NaN
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['typhoid fever'] = pd.to_numeric(df['typhoid fever'], errors='coerce')

# Drop rows with missing values after conversion
df.dropna(subset=['typhus', 'typhoid fever'], inplace=True)

# Calculate correlation coefficient
correlation = df['typhus'].corr(df['typhoid fever'])

print(f"Final Answer: {correlation:.2f}")