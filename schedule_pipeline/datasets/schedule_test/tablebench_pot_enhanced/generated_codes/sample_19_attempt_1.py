import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where industry is 'oil'
oil_companies = df[df['industry'] == 'oil']
# Sum the revenue for oil companies
total_revenue_oil = oil_companies['revenue (millions)'].sum()
print(f"Final Answer: {total_revenue_oil}")