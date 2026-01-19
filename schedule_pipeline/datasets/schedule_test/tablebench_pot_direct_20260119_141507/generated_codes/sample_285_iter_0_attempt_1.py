import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where industry is 'banking'
banking_companies = df[df['industry'] == 'banking']

# Select the top 5 (first 5 rows) based on rank
top_5_banking = banking_companies.head(5)

# Calculate the average revenue of these companies
average_revenue = top_5_banking['revenue (millions)'].mean()
print(f"Final Answer: {average_revenue:.1f}")