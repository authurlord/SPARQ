import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert sales column to numeric, coercing errors to NaN
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')

# Filter for oil and gas industry
oil_gas_df = df[df['industry'] == 'oil and gas'].copy()

# Sort by sales in descending order and take top 5
top_5_oil_gas = oil_gas_df.nlargest(5, 'sales (billion )')

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_5_oil_gas['company'], top_5_oil_gas['sales (billion )'], color='skyblue')
plt.title('Sales of Top 5 Oil and Gas Companies')
plt.xlabel('Company')
plt.ylabel('Sales (billion USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final Answer: The bar chart has been created showing the sales of the top 5 oil and gas companies.
Final Answer: bar_chart