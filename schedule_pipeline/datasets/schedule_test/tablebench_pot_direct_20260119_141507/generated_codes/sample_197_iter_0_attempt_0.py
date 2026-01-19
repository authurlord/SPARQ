import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between tourist arrivals and tourism receipts
correlation = df['internl tourist arrivals 2011 (x1000)'].corr(df['internl tourism receipts 2011 (million usd)'])
print(f"Final Answer: {correlation:.2f}")