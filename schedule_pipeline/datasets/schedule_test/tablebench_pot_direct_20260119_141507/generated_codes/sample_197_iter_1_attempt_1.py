import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns with correct names
arrivals = df['internl tourist arrivals 2011 (x1000)']
receipts = df['internl tourism receipts 2011 (million usd)']

# Calculate the correlation between arrivals and receipts
correlation = arrivals.corr(receipts)

print(f"Final Answer: {correlation:.2f}")