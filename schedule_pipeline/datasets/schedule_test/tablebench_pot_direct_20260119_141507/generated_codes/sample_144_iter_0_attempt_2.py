import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter companies in the oil and gas industry
oil_gas_companies = df[df['industry'] == 'oil and gas']

# Sort by sales in descending order and take top 5
top_5_oil_gas = oil_gas_companies.sort_values(by='sales (billion )', ascending=False).head(5)

# Extract company names and sales
companies = top_5_oil_gas['company']
sales = top_5_oil_gas['sales (billion )'].astype(float)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(companies, sales, color='skyblue')
plt.title('Sales of Top 5 Oil and Gas Companies')
plt.xlabel('Company')
plt.ylabel('Sales (billion USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final Answer: The bar chart has been created and displayed.
Final Answer: bar chart created