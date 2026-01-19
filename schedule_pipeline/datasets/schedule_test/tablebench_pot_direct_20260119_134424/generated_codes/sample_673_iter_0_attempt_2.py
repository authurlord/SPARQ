import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'line length (kilometres)' column by removing commas and converting to float
df['line length (kilometres)'] = df['line length (kilometres)'].str.replace(',', '').astype(float)
# Clean the 'annual ridership (1998)' column by removing commas and converting to float
df['annual ridership (1998)'] = df['annual ridership (1998)'].str.replace(',', '').astype(float)
# Drop the 'totals :' row as it is not a line
df = df[df['line'] != 'totals :']
# Calculate the correlation coefficient
correlation = df['line length (kilometres)'].corr(df['annual ridership (1998)'])
print(f"Final Answer: {correlation:.4f}")