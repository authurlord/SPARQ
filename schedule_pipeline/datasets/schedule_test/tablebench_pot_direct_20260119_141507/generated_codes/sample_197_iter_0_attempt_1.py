import pandas as pd

df = pd.read_csv('table.csv')
# Select the relevant columns
arrivals = df['internl tourist arrivals 2011 (x1000)']
receipts = df['internl tourism receipts 2011 (million usd)']

# Calculate the correlation coefficient
correlation = arrivals.corr(receipts)
print(f"Final Answer: {correlation:.2f}")