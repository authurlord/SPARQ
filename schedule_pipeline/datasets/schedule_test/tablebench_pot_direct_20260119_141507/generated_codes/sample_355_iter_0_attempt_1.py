import pandas as pd

df = pd.read_csv('table.csv')
# Filter banks with total assets > 10,000 million TL
filtered_banks = df[df['Total Assets (million TL) As of 30 September 2012'] > 10000]
count = len(filtered_banks)
print(f"Final Answer: {count}")