import pandas as pd

df = pd.read_csv('table.csv')
# Filter banks with total assets > 10,000 million TL
count_high_assets = df[df['Total Assets (million TL) As of 30 September 2012'] > 10000].shape[0]
print(f"Final Answer: {count_high_assets}")