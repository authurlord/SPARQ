import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where industry is 'banking'
banking_companies = df[df['industry'] == 'banking']
# Convert sales to float and sum
total_sales = banking_companies['sales (billion )'].astype(float).sum()
print(f"Final Answer: {total_sales:.2f}")