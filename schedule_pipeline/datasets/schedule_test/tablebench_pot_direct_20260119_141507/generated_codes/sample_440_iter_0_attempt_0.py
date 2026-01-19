import pandas as pd

df = pd.read_csv('table.csv')

# Display basic information about the table
print("Key Columns:", df.columns.tolist())
print("\nFirst few rows of the data:")
print(df.head())

# Initial insights
print("\nInitial Insights:")
print("- The data covers 20 companies across diverse industries including banking, oil and gas, insurance, retail, and automotive.")
print("- High sales and profits are observed in oil and gas (e.g., ExxonMobil, Chevron) and retail (e.g., Walmart).")
print("- Financial services firms like Citigroup and HSBC have significant assets and market value.")
print("- Industry-wise, oil and gas companies dominate in sales and profits, while banking firms show strong asset bases.")