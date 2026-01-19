import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' and 'typhoid fever' to numeric, handling non-numeric values like 'avg'
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['typhoid fever'] = pd.to_numeric(df['typhoid fever'], errors='coerce')

# Drop rows with NaN values after conversion
df.dropna(subset=['typhus', 'typhoid fever'], inplace=True)

# Calculate correlation coefficient
correlation = df['typhus'].corr(df['typhoid fever'])

# Print the correlation value
print(f"Final Answer: {correlation:.2f}")