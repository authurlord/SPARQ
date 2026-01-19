import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert 'sales (billion )' to numeric, coercing errors to NaN
df['sales (billion )'] = pd.to_numeric(df['sales (billion )'], errors='coerce')

# Filter companies in the 'oil and gas' industry
oil_gas_df = df[df['industry'] == 'oil and gas'].copy()

# Sort by sales in descending order and get top 5
top_5_oil_gas = oil_gas_df.nlargest(5, 'sales (billion )')

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_5_oil_gas['company'], top_5_oil_gas['sales (billion )'], color='skyblue')
plt.title('Top 5 Oil and Gas Companies by Sales')
plt.xlabel('Company')
plt.ylabel('Sales (billion)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the sales values for clarity (optional output)
print(top_5_oil_gas[['company', 'sales (billion )']])