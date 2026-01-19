import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'number of branches' and 'total assets' to numeric
df['# of Branches As of 30 September 2012'] = df['# of Branches As of 30 September 2012'].str.replace(',', '').astype(float)
df['Total Assets (million TL) As of 30 September 2012'] = df['Total Assets (million TL) As of 30 September 2012'].str.replace(',', '').astype(float)

# Calculate correlation coefficient
correlation = df['# of Branches As of 30 September 2012'].corr(df['Total Assets (million TL) As of 30 September 2012'])

print(f"Final Answer: {correlation:.4f}")