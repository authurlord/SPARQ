import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'number of branches' and 'total assets' columns
df['# of Branches As of 30 September 2012'] = df['# of Branches As of 30 September 2012'].str.replace(',', '').astype(int)
df['Total Assets (million TL) As of 30 September 2012'] = df['Total Assets (million TL) As of 30 September 2012'].str.replace(',', '').astype(int)

# Calculate correlation coefficient
correlation = df['# of Branches As of 30 September 2012'].corr(df['Total Assets (million TL) As of 30 September 2012'])

print(f"Final Answer: {correlation:.4f}")