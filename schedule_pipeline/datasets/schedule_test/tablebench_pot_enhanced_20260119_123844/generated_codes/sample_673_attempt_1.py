import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'line length (kilometres)' and 'annual ridership (1998)' to numeric
df['line length (kilometres)'] = df['line length (kilometres)'].str.replace(',', '').astype(float)
df['annual ridership (1998)'] = df['annual ridership (1998)'].str.replace(',', '').astype(float)

# Drop the 'totals :' row as it's a summary and not a line
df_clean = df[df['line'] != 'totals :']

# Calculate correlation coefficient
correlation = df_clean['line length (kilometres)'].corr(df_clean['annual ridership (1998)'])

print(f"Final Answer: {correlation:.4f}")