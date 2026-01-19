import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'sales (billion )' to numeric, coercing errors to NaN
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')

# Filter companies in the 'oil and gas' industry
oil_and_gas = df[df['industry'] == 'oil and gas']

# Sort by sales in descending order and take top 5
top_5_oil_gas = oil_and_gas.nlargest(5, 'sales (billion )')

# Extract company names and sales for plotting
companies = top_5_oil_gas['company']
sales = top_5_oil_gas['sales (billion )']

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(companies, sales, color='skyblue')
plt.title('Top 5 Oil and Gas Companies by Sales')
plt.xlabel('Company')
plt.ylabel('Sales (billion USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the sales values for reference
print(f"Final Answer: {list(sales)}")