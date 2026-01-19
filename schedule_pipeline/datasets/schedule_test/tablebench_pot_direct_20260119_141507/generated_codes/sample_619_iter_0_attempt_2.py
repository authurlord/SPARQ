import pandas as pd

df = pd.read_csv('table.csv')
# Filter for banking industry
banking_df = df[df['industry'] == 'banking']
# Sort by sales in descending order and take top 3
top_3_banking = banking_df.nlargest(3, 'sales (billion )')
# Sum the sales of these top 3
total_sales = top_3_banking['sales (billion )'].sum()
print(f"Final Answer: {total_sales}")