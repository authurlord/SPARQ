import pandas as pd

df = pd.read_csv('table.csv')
# Filter companies in the 'oil' industry and sum their revenue
oil_revenue = df[df['industry'] == 'oil']['revenue (millions)'].sum()
print(f"Final Answer: {oil_revenue}")