import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert 'Total Assets (million TL) As of 30 September 2012' to numeric
df['Total Assets (million TL) As of 30 September 2012'] = df['Total Assets (million TL) As of 30 September 2012'].str.replace(',', '').astype(float)
# Count banks with assets > 10,000 million TL
count_banks = (df['Total Assets (million TL) As of 30 September 2012'] > 10000).sum()
print(f"Final Answer: {count_banks}")