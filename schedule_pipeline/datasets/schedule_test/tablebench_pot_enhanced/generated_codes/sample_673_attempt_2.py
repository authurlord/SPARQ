import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert the columns to numeric
df['line length (kilometres)'] = df['line length (kilometres)'].str.replace(',', '').astype(float)
df['annual ridership (1998)'] = df['annual ridership (1998)'].str.replace(',', '').astype(float)

# Calculate correlation coefficient
correlation = df['line length (kilometres)'].corr(df['annual ridership (1998)'])
print(f"Final Answer: {correlation:.4f}")