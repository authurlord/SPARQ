import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry
banking_companies = df[df['industry'] == 'banking']
# Sort by sales in descending order and take top 3
top_3_bankers = banking_companies.nlargest(3, 'sales (billion )')
# Calculate total sales
total_sales = top_3_bankers['sales (billion )'].astype(float).sum()
print(f"Final Answer: {total_sales:.2f}")