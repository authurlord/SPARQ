import pandas as pd

df = pd.read_csv('table.csv')
# Filter for oil industry
oil_companies = df[df['industry'] == 'oil']
# Sum the revenue
total_revenue_oil = oil_companies['revenue (millions)'].sum()
print(f"Final Answer: {total_revenue_oil}")