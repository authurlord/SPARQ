import pandas as pd

df = pd.read_csv('table.csv')
# Filter for companies in the 'banking' industry
banking_companies = df[df['industry'] == 'banking']
# Convert sales to float and sum the top 3 (by rank)
total_sales = banking_companies['sales (billion )'].astype(float).sum()
print(f"Final Answer: {total_sales:.2f}")