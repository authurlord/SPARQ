import pandas as pd

df = pd.read_csv('table.csv')

# Display key columns and initial insights
print("Key Columns: company, industry, sales (billion ), profits (billion ), assets (billion ), market value (billion )")
print("\nInitial Insights:")
print("- The data covers top companies across diverse industries such as banking, oil and gas, insurance, and retail.")
print("- Oil and gas companies (e.g., ExxonMobil, Chevron) have the highest sales and profits.")
print("- Banking sector companies (e.g., Citigroup, HSBC) show strong sales and assets.")
print("- Walmart has the highest sales among retail companies and is a major player in retail.")
print("- Market value and assets vary widely, indicating differences in company size and performance.")