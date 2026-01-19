import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter companies in the 'oil and gas' industry
oil_gas_df = df[df['industry'] == 'oil and gas']

# Sort by sales (billion) in descending order and take top 5
top_5_oil_gas = oil_gas_df.nlargest(5, 'sales (billion )')

# Prepare data for bar chart
companies = top_5_oil_gas['company']
sales = top_5_oil_gas['sales (billion )'].astype(float)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(companies, sales, color='skyblue')
plt.title('Sales of Top 5 Companies in Oil and Gas Industry')
plt.xlabel('Company')
plt.ylabel('Sales (billion USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the sales values for reference (optional)
print(f"Final Answer: {list(sales)}")