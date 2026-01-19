import pandas as pd

df = pd.read_csv('table.csv')
# Clean and convert the 'Total Assets (million TL) As of 30 September 2012' column
df['Total Assets (million TL) As of 30 September 2012'] = df['Total Assets (million TL) As of 30 September 2012'].str.replace(',', '').astype(int)
# Count banks with assets > 10,000 million TL
count_banks = df[df['Total Assets (million TL) As of 30 September 2012'] > 10000].shape[0]
print(f"Final Answer: {count_banks}")