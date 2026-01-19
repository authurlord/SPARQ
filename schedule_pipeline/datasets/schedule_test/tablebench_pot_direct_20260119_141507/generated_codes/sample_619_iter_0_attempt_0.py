import pandas as pd

df = pd.read_csv('table.csv')

# Filter banking industry companies
banking_companies = df[df['industry'] == 'banking']

# Sort by sales in descending order and take top 3
top_3_banking = banking_companies.nlargest(3, 'sales (billion )')

# Clean and convert sales column to float (remove commas and spaces)
def clean_sales(sales_str):
    return float(sales_str.replace(',', '').replace(' ', ''))

total_sales = top_3_banking['sales (billion )'].apply(clean_sales).sum()

print(f"Final Answer: {total_sales:.2f}")