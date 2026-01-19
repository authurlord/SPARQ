import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' and 'typhoid fever' columns to numeric
df['typhus'] = pd.to_numeric(df['typhus'])
df['typhoid fever'] = pd.to_numeric(df['typhoid fever'])

# Calculate correlation coefficient
correlation = df['typhus'].corr(df['typhoid fever'])

# Output the result
print(f"Final Answer: {correlation:.2f}")