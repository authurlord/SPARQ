import pandas as pd

df = pd.read_csv('table.csv')
# Convert the Total Assets column to numeric by removing commas
df['Total Assets (million TL) As of 30 September 2012'] = df['Total Assets (million TL) As of 30 September 2012'].str.replace(',', '').astype(int)
# Count banks with total assets > 10,000 million TL
count = df[df['Total Assets (million TL) As of 30 September 2012'] > 10000].shape[0]
print(f"Final Answer: {count}")