import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry
banking_companies = df[df['industry'] == 'banking']
# Sort by sales in descending order and get top 3
top_3_bank_sales = banking_companies.nlargest(3, 'sales (billion )')['sales (billion )'].astype(float).sum()
print(f"Final Answer: {top_3_bank_sales:.2f}")